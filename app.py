import streamlit as st
import pymupdf  # PyMuPDF
import ollama
from ollama import AsyncClient
import asyncio
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import base64
import gc
import os
import json
import uuid
import shutil
import logging
import sys
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import queue
import threading
import time

# Configurare sistem de logare pentru a scrie atat in fisier cat si pe ecran
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Activam logurile serverului web intern (Tornado) pentru a vedea fix cand pica conexiunea cu Cloudflare
logging.getLogger("tornado.access").setLevel(logging.INFO)
logging.getLogger("tornado.application").setLevel(logging.INFO)
logging.getLogger("tornado.general").setLevel(logging.INFO)

st.set_page_config(page_title="Multi-Document RAG Evaluator", layout="wide")

VISION_MODEL = "llama3.2-vision:latest"
SYNTHESIS_MODEL = "mistral:latest"
EMBEDDING_MODEL = "nomic-embed-text"

DOCS_DIR = "data/documents"
EVALS_DIR = "data/evaluations"
CHROMA_DIR = "data/chroma_db"

for d in [DOCS_DIR, EVALS_DIR, CHROMA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
        
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
                embeddings.append(res["embedding"])
            except Exception as e:
                # Fallback to zero vector if model is missing, though we should warn user
                st.error(f"Embedding failed. Did you run 'ollama pull {EMBEDDING_MODEL}'? Error: {e}")
                embeddings.append([0]*768)
        return embeddings

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(
    name="pdf_collection",
    embedding_function=OllamaEmbeddingFunction()
)

@st.cache_resource
def get_task_queue():
    q = queue.Queue()
    def worker():
        while True:
            task = q.get()
            if task is None:
                break
            try:
                process_evaluation_task(task)
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
            finally:
                q.task_done()
                
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    if os.path.exists(EVALS_DIR):
        for folder in os.listdir(EVALS_DIR):
            meta_path = os.path.join(EVALS_DIR, folder, "eval_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("status") == "In Progress":
                        logger.info(f"Reluam task-ul neterminat: {meta['eval_id']}")
                        q.put({
                            "eval_id": meta["eval_id"],
                            "prompt": meta["prompt"],
                            "docs": meta.get("docs", [])
                        })
                except Exception as e:
                    logger.error(f"Eroare la incarcarea persistentei: {e}")
                    
    return q

task_queue = get_task_queue()

def get_base64_image(page):
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")

def extract_text_with_vision(base64_image):
    prompt = "Extract all text and data from this image cleanly. If there are tables, format them using Markdown. If there are charts, describe them in detail."
    try:
        response = ollama.chat(
            model=VISION_MODEL, 
            messages=[{'role': 'user', 'content': prompt, 'images': [base64_image]}],
            options={"temperature": 0.0, "num_predict": 4096}
        )
        if 'error' in response:
            raise Exception(response['error'])
        return response.get('message', {}).get('content', '')
    except Exception as e:
        error_msg = f"Vision extraction failed: {e}"
        logger.error(f"[{VISION_MODEL}] ERROR: {error_msg}")
        st.error(error_msg)
        return ""

def generate_rag_answer(criteria, context_chunks):
    system_instruction = (
        "You are an expert grant reviewer and professional analyst.\n"
        "Your task is to provide a SINGLE, cohesive, and comprehensive response that synthesizes information across all provided document excerpts.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. SYNTHESIS: DO NOT evaluate each excerpt separately. Read all excerpts and synthesize the findings into ONE unified, coherent evaluation.\n"
        "2. NO REDUNDANCY: DO NOT repeat the same points. If multiple excerpts mention the same thing, combine them into a single point.\n"
        "3. PROFESSIONAL FORMATTING: You MUST format the output professionally using clean Markdown. Use headings (##), bullet points for lists, and **bold text** for emphasis. YOU MUST include blank lines between paragraphs and sections to make it highly readable and prevent it from looking like a block of text.\n"
        "4. STRICT ACCURACY: Answer the user's prompt based ONLY on the provided excerpts."
    )
    
    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = f"Please read the following Excerpts and provide ONE unified answer to the User Prompt.\n\nUser Prompt:\n{criteria}\n\nExcerpts:\n{context_str}"
    
    try:
        import time
        start_time = time.time()
        logger.info(f"[{SYNTHESIS_MODEL}] A inceput generarea raspunsului (asta poate dura mult)...")
        response = ollama.chat(
            model=SYNTHESIS_MODEL, 
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': prompt}
            ],
            options={"temperature": 0.0, "num_predict": 4096}
        )
        duration = time.time() - start_time
        logger.info(f"[{SYNTHESIS_MODEL}] Raspuns primit cu succes in {duration:.2f} secunde.")
        if 'error' in response:
            raise Exception(response['error'])
        return response.get('message', {}).get('content', '')
    except Exception as e:
        error_msg = f"Model error during synthesis: {e}"
        logger.error(f"[{SYNTHESIS_MODEL}] ERROR: {error_msg}")
        raise Exception(error_msg)

# --- DOCUMENT INGESTION (STAGE 1) ---

def ingest_document(uploaded_file):
    doc_id = str(uuid.uuid4())
    doc_dir = os.path.join(DOCS_DIR, doc_id)
    os.makedirs(doc_dir)
    
    pdf_path = os.path.join(doc_dir, "document.pdf")
    pdf_bytes = uploaded_file.getvalue()
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    
    meta = {
        "doc_id": doc_id,
        "filename": uploaded_file.name,
        "total_pages": total_pages,
        "processed_pages": 0,
        "ingestion_status": "Processing"
    }
    with open(os.path.join(doc_dir, "doc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    return doc_id, doc_dir, pdf_path, total_pages

def process_document_pages(doc_id, doc_dir, pdf_path, total_pages, filename):
    doc = pymupdf.open(pdf_path)
    progress_bar = st.progress(0)
    status_text = st.empty()
    successfully_imported = 0
    
    for i in range(total_pages):
        status_text.text(f"Ingesting page {i+1}/{total_pages}...")
        page = doc.load_page(i)
        native_text = page.get_text()
        
        # SMART ROUTING
        text_len = len(native_text.strip())
        num_images = len(page.get_images())
        
        if (num_images > 0 and text_len < 300) or text_len <= 50:
            base64_image = get_base64_image(page)
            final_text = extract_text_with_vision(base64_image)
            del base64_image
        else:
            final_text = native_text
            
        # Store in ChromaDB
        if final_text.strip():
            collection.add(
                documents=[final_text],
                metadatas=[{"doc_id": doc_id, "filename": filename, "page": i+1}],
                ids=[f"{doc_id}_page_{i+1}"]
            )
            successfully_imported += 1
            
        # Update metadata to track real-time partial progress
        meta_path = os.path.join(doc_dir, "doc_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["processed_pages"] = successfully_imported
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            
        del page
        del native_text
        gc.collect()
        progress_bar.progress((i + 1) / total_pages)
        
    doc.close()
    status_text.text("Ingestion Complete!")
    
    # Update metadata
    meta_path = os.path.join(doc_dir, "doc_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["ingestion_status"] = "Ready"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

def load_documents():
    docs = []
    if os.path.exists(DOCS_DIR):
        for folder in os.listdir(DOCS_DIR):
            meta_path = os.path.join(DOCS_DIR, folder, "doc_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        docs.append(json.load(f))
                except Exception:
                    pass
    return docs

def delete_document(doc_id):
    # Remove from File System
    doc_dir = os.path.join(DOCS_DIR, doc_id)
    if os.path.exists(doc_dir):
        shutil.rmtree(doc_dir)
    # Remove from ChromaDB
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception as e:
        pass

# --- EVALUATIONS (STAGE 2) ---

def process_evaluation_task(task):
    eval_id = task["eval_id"]
    prompt = task["prompt"]
    selected_docs_meta = task["docs"]
    eval_dir = os.path.join(EVALS_DIR, eval_id)
    
    logger.info(f"Processing task for eval_id: {eval_id}")
    try:
        selected_doc_ids = [d["doc_id"] for d in selected_docs_meta]
        results = collection.query(
            query_texts=[prompt],
            n_results=15,
            where={"doc_id": {"$in": selected_doc_ids}}
        )
        
        retrieved_chunks = results['documents'][0]
        retrieved_meta = results['metadatas'][0]
        
        formatted_chunks = []
        for chunk, m in zip(retrieved_chunks, retrieved_meta):
            formatted_chunks.append(f"**Source: {m['filename']} (Page {m['page']})**\n{chunk}")
            
        final_answer = generate_rag_answer(prompt, formatted_chunks)
        
        final_report = f"# Evaluation Results\n\n**Prompt:** {prompt}\n\n## Answer\n\n{final_answer}\n\n## Sources Used\n\n"
        for m in retrieved_meta:
            final_report += f"- {m['filename']} (Page {m['page']})\n"
            
        with open(os.path.join(eval_dir, "final_report.md"), "w", encoding="utf-8") as f:
            f.write(final_report)
            
        meta_path = os.path.join(eval_dir, "eval_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "Completed"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            
        logger.info(f"Task completed for eval_id: {eval_id}")
    except Exception as e:
        logger.error(f"Error processing evaluation {eval_id}: {e}")
        meta_path = os.path.join(eval_dir, "eval_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["status"] = "Error"
                meta["error_message"] = str(e)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4)
            except:
                pass

def create_evaluation(prompt, selected_docs_meta):
    eval_id = str(uuid.uuid4())
    eval_dir = os.path.join(EVALS_DIR, eval_id)
    os.makedirs(eval_dir)
    
    meta = {
        "eval_id": eval_id,
        "prompt": prompt,
        "docs": selected_docs_meta,
        "status": "In Progress" 
    }
    
    with open(os.path.join(eval_dir, "eval_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
        
    task_queue.put({
        "eval_id": eval_id,
        "prompt": prompt,
        "docs": selected_docs_meta
    })
    
    return eval_id

def load_evaluations():
    evals = []
    if os.path.exists(EVALS_DIR):
        for folder in os.listdir(EVALS_DIR):
            meta_path = os.path.join(EVALS_DIR, folder, "eval_meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        evals.append(json.load(f))
                except Exception:
                    pass
    return evals

def delete_evaluation(eval_id):
    eval_dir = os.path.join(EVALS_DIR, eval_id)
    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)

def get_final_report(eval_id):
    eval_dir = os.path.join(EVALS_DIR, eval_id)
    report_path = os.path.join(eval_dir, "final_report.md")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

# --- UI ---

def main():
    with st.sidebar:
        st.header("System")
        if st.button("Restart Server (Clear Cache)", help="Forces the Python server to restart. If you use run_app.bat, it will automatically come back online."):
            os._exit(0)

    st.title("Multi-Document Hybrid RAG Evaluator")

    tab1, tab2 = st.tabs(["📚 Document Library (Ingestion)", "⚙️ Evaluations (Q&A)"])
    
    with tab1:
        render_document_library()
        
    with tab2:
        render_evaluations_dashboard()

def render_document_library():
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    st.header("Upload & Ingest New Document")
    st.write("Documents uploaded here will be processed page-by-page. Native text is extracted instantly. Scanned pages are read by `llama3.2-vision`. Everything is indexed into the local Vector Database.")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")
    if st.button("Ingest Document"):
        if uploaded_file:
            doc_id, doc_dir, pdf_path, total_pages = ingest_document(uploaded_file)
            st.info(f"Starting ingestion for {uploaded_file.name} ({total_pages} pages). This may take a while if vision extraction is heavily used...")
            process_document_pages(doc_id, doc_dir, pdf_path, total_pages, uploaded_file.name)
            st.success("Document ingested and vectorized successfully!")
            st.session_state.uploader_key += 1
            st.rerun()
        else:
            st.warning("Please select a PDF file first.")
            
    st.divider()
    st.header("Available Indexed Documents")
    docs = load_documents()
    if not docs:
        st.info("The library is empty. Upload some PDFs above.")
    else:
        for doc in docs:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            col1.write(f"📄 **{doc['filename']}**")
            col2.write(f"Pages: {doc.get('processed_pages', 0)} / {doc['total_pages']} imported")
            status = doc.get('ingestion_status', 'Unknown')
            col3.write(f"Status: {status}")
            if col4.button("Delete", key=f"del_doc_{doc['doc_id']}"):
                delete_document(doc['doc_id'])
                st.rerun()

def render_evaluations_dashboard():
    docs = load_documents()
    ready_docs = [d for d in docs if d.get('ingestion_status') == 'Ready']
    
    st.header("Ask Questions Across Documents")
    
    doc_options = {d['doc_id']: f"{d['filename']} ({d['total_pages']} pages)" for d in ready_docs}
    selected_doc_ids = st.multiselect("Select Documents to Query", options=list(doc_options.keys()), format_func=lambda x: doc_options[x])
    
    criteria = st.text_area("Question / Evaluation Prompt", height=100, placeholder="e.g., Extract all financial figures related to Q3 revenue across the selected documents.")
    
    if st.button("Generate Answer"):
        if selected_doc_ids and criteria:
            selected_docs_meta = [d for d in ready_docs if d['doc_id'] in selected_doc_ids]
            eval_id = create_evaluation(criteria, selected_docs_meta)
            st.session_state.view_eval_id = eval_id
            
            status_container = st.empty()
            dots = 0
            try:
                while True:
                    meta_path = os.path.join(EVALS_DIR, eval_id, "eval_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        
                        if meta.get("status") == "Completed":
                            status_container.success("Answer generated successfully!")
                            break
                        elif meta.get("status") == "Error":
                            status_container.error(f"Error during generation: {meta.get('error_message', 'Unknown')}")
                            break
                    
                    dots = (dots + 1) % 4
                    status_container.info(f"Task queued and processing in background{'.' * dots}\n(Cloudflare connection is kept alive)")
                    time.sleep(2)
            except Exception as e:
                logger.warning(f"UI waiting loop interrupted (client likely disconnected): {e}")
                
        else:
            st.warning("Please select at least one document and enter a question.")
            
    st.divider()
    
    if getattr(st.session_state, 'view_eval_id', None):
        st.subheader("Latest Result")
        report = get_final_report(st.session_state.view_eval_id)
        if report:
            st.markdown(report)
        else:
            st.warning("Report is not ready or encountered an error.")
        st.divider()

    st.header("Past Q&A Sessions")
    evals = load_evaluations()
    if not evals:
        st.info("No past evaluations.")
    else:
        for ev in reversed(evals):
            status = ev.get('status', 'Completed')
            status_icon = "⏳" if status == "In Progress" else "✅" if status == "Completed" else "❌"
            with st.expander(f"{status_icon} Q: {ev['prompt'][:50]}..."):
                doc_names = ", ".join([d['filename'] for d in ev['docs']])
                st.caption(f"Status: **{status}** | Sources: {doc_names}")
                
                if status == "Completed":
                    report = get_final_report(ev['eval_id'])
                    if report:
                        st.markdown(report)
                elif status == "Error":
                    st.error(ev.get("error_message", "Unknown error."))
                else:
                    st.info("Task is currently in progress. Please check back later.")
                    
                if st.button("Delete Log", key=f"del_{ev['eval_id']}"):
                    delete_evaluation(ev['eval_id'])
                    st.rerun()

if __name__ == "__main__":
    main()
