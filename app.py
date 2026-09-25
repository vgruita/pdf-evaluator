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

def llm_chat(messages, model=SYNTHESIS_MODEL, temperature=0.0):
    try:
        import time
        start_time = time.time()
        logger.info(f"[{model}] Starting LLM generation...")
        response = ollama.chat(
            model=model, 
            messages=messages,
            options={"temperature": temperature, "num_predict": 4096}
        )
        duration = time.time() - start_time
        logger.info(f"[{model}] Response received in {duration:.2f}s")
        
        if 'error' in response:
            raise Exception(response['error'])
        return response.get('message', {}).get('content', '')
    except Exception as e:
        error_msg = f"Model error: {e}"
        logger.error(f"[{model}] ERROR: {error_msg}")
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
    user_prompt = task["prompt"]
    selected_docs_meta = task["docs"]
    eval_dir = os.path.join(EVALS_DIR, eval_id)
    
    def log_and_update(msg):
        logger.info(msg)
        meta_path = os.path.join(eval_dir, "eval_meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["current_step"] = msg
            if "logs" not in meta:
                meta["logs"] = []
            meta["logs"].append(msg)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
        except Exception:
            pass
            
    log_and_update(f"Processing complex Multi-Step evaluation for: {eval_id}")
    try:
        selected_doc_ids = [d["doc_id"] for d in selected_docs_meta]
        
        # 1.1 EXTRAGERE CRITERII (Breakdown)
        log_and_update("Step 1.1: Breakdown prompt into criteria...")
        breakdown_msg = [
            {'role': 'system', 'content': 'You are a logical analyst. Break down the provided evaluation rubric into a list of distinct sections/criteria that need to be evaluated. Return them as a numbered list.'},
            {'role': 'user', 'content': f"Rubric:\n{user_prompt}"}
        ]
        criteria_list_text = llm_chat(breakdown_msg, temperature=0.1)
        
        criteria_items = [c.strip() for c in criteria_list_text.split('\n') if c.strip() and c.strip()[0].isdigit()]
        if not criteria_items:
            criteria_items = [user_prompt]
            
        verified_main_conclusions = []
        all_retrieved_meta = []
        
        steps_dir = os.path.join(eval_dir, "steps")
        if not os.path.exists(steps_dir):
            os.makedirs(steps_dir)
            
        for c_idx, main_criterion in enumerate(criteria_items, 1):
            log_and_update(f"Processing MAIN criterion: {main_criterion[:60]}...")
            
            # 1.2 Breakdown Main Criterion into sub-criteria
            log_and_update(f"Step 1.2: Breakdown MAIN criterion into sub-criteria...")
            sub_breakdown_msg = [
                {'role': 'system', 'content': 'You are a logical analyst. Break down the provided complex evaluation criterion into smaller, distinct sub-criteria or steps to check. Return them as a numbered list. If the criterion is very simple, just return it as a single item list.'},
                {'role': 'user', 'content': f"Criterion:\n{main_criterion}"}
            ]
            sub_criteria_list_text = llm_chat(sub_breakdown_msg, temperature=0.1)
            sub_criteria_items = [c.strip() for c in sub_criteria_list_text.split('\n') if c.strip() and c.strip()[0].isdigit()]
            if not sub_criteria_items:
                sub_criteria_items = [main_criterion]
                
            verified_sub_conclusions = []
            
            for s_idx, sub_crit in enumerate(sub_criteria_items, 1):
                log_and_update(f"Processing SUB-criterion: {sub_crit[:60]}...")
                
                results = collection.query(
                    query_texts=[sub_crit],
                    n_results=7,
                    where={"doc_id": {"$in": selected_doc_ids}}
                )
                chunks = results['documents'][0]
                metas = results['metadatas'][0]
                all_retrieved_meta.extend(metas)
                
                context_str = "\n---\n".join([f"(Source: {m['filename']}, Page {m['page']}) {c}" for c, m in zip(chunks, metas)])
                
                # 1.3 INITIAL CONCLUSION FOR SUB-CRITERION
                eval_msg = [
                    {'role': 'system', 'content': 'Evaluate the given sub-criterion based strictly on the provided document excerpts.'},
                    {'role': 'user', 'content': f"Sub-criterion to evaluate: {sub_crit}\n\nExcerpts:\n{context_str}"}
                ]
                initial_conclusion = llm_chat(eval_msg, temperature=0.0)
                
                # 1.4 VERIFICATION FOR SUB-CRITERION
                critique_msg = [
                    {'role': 'system', 'content': 'Identify any missing information or weaknesses mentioned in the conclusion. If none, reply with exactly "NONE".'},
                    {'role': 'user', 'content': f"Conclusion: {initial_conclusion}"}
                ]
                weaknesses = llm_chat(critique_msg, temperature=0.0)
                
                verify_context = ""
                if "NONE" not in weaknesses.upper() and len(weaknesses) > 10:
                    log_and_update(f"Possible gaps found in sub-criterion. Re-querying...")
                    verify_results = collection.query(
                        query_texts=[weaknesses],
                        n_results=4,
                        where={"doc_id": {"$in": selected_doc_ids}}
                    )
                    verify_chunks = verify_results['documents'][0]
                    verify_context = "\n---\n".join(verify_chunks)
                    
                    resolve_msg = [
                        {'role': 'system', 'content': 'You previously wrote a conclusion that found some weaknesses. Here is new context. If the new context resolves the weaknesses, rewrite the conclusion to fix the mistakes. If the weaknesses are still true, output the original conclusion.'},
                        {'role': 'user', 'content': f"Original Conclusion: {initial_conclusion}\n\nWeaknesses Found: {weaknesses}\n\nNew Context: {verify_context}"}
                    ]
                    final_sub_conclusion = llm_chat(resolve_msg, temperature=0.0)
                else:
                    final_sub_conclusion = initial_conclusion
                    
                verified_sub_conclusions.append(f"- Sub-evaluation for '{sub_crit}':\n{final_sub_conclusion}")
                
                # Save Sub-Criterion Trace
                sub_trace_file = os.path.join(steps_dir, f"criterion_{c_idx}_sub_{s_idx}.md")
                with open(sub_trace_file, "w", encoding="utf-8") as f:
                    f.write(f"## Main Criterion\n{main_criterion}\n\n")
                    f.write(f"### Sub-Criterion\n{sub_crit}\n\n")
                    f.write(f"### Initial Conclusion\n{initial_conclusion}\n\n")
                    f.write(f"### Identified Weaknesses\n{weaknesses}\n\n")
                    if verify_context:
                        f.write(f"### Verification Context Retrieved\n{verify_context}\n\n")
                    f.write(f"### Final Sub-Conclusion\n{final_sub_conclusion}\n")
                
            # 1.5 SYNTHESIZE SUB-CRITERIA INTO MAIN CRITERION
            log_and_update(f"Step 1.5: Synthesizing sub-criteria for '{main_criterion[:40]}'...")
            all_sub_text = "\n\n".join(verified_sub_conclusions)
            synth_main_msg = [
                {'role': 'system', 'content': 'You are a professional grant reviewer. Synthesize the provided sub-evaluations into one cohesive, comprehensive answer that directly addresses the Main Criterion. Do not list the sub-criteria, write a unified evaluation.'},
                {'role': 'user', 'content': f"Main Criterion:\n{main_criterion}\n\nSub-evaluations to synthesize:\n{all_sub_text}"}
            ]
            main_conclusion = llm_chat(synth_main_msg, temperature=0.0)
            verified_main_conclusions.append(f"### Evaluation for:\n{main_criterion}\n\n{main_conclusion}\n")
            
            # Save Main Criterion Trace
            main_trace_file = os.path.join(steps_dir, f"criterion_{c_idx}_summary.md")
            with open(main_trace_file, "w", encoding="utf-8") as f:
                f.write(f"## Main Criterion\n{main_criterion}\n\n")
                f.write(f"### Sub-evaluations Provided\n{all_sub_text}\n\n")
                f.write(f"### Synthesized Main Conclusion\n{main_conclusion}\n")
            
        # 1.6 FINAL GLOBAL SYNTHESIS
        log_and_update("Step 1.6: Final global synthesis...")
        all_conclusions_text = "\n".join(verified_main_conclusions)
        synthesis_msg = [
            {'role': 'system', 'content': 'You are a professional grant reviewer. Your task is to take the provided verified evaluations and format them EXACTLY as requested in the User Formatting Prompt & Constraints (use exact Markdown headers, numbers, etc). Do not skip any sections.'},
            {'role': 'user', 'content': f"User Formatting Prompt & Constraints:\n{user_prompt}\n\nVerified Evaluations to Format:\n{all_conclusions_text}"}
        ]
        final_answer = llm_chat(synthesis_msg, temperature=0.0)
        
        unique_sources = []
        seen = set()
        for m in all_retrieved_meta:
            sig = f"{m['filename']} (Page {m['page']})"
            if sig not in seen:
                seen.add(sig)
                unique_sources.append(sig)
                
        final_report = f"# Evaluation Results\n\n## Answer\n\n{final_answer}\n\n## Sources Used\n\n"
        for s in unique_sources:
            final_report += f"- {s}\n"
            
        with open(os.path.join(eval_dir, "final_report.md"), "w", encoding="utf-8") as f:
            f.write(final_report)
            
        meta_path = os.path.join(eval_dir, "eval_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "Completed"
        meta["current_step"] = "Finished"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)
            
        log_and_update(f"Task completed successfully for eval_id: {eval_id}")
    except Exception as e:
        logger.error(f"Error processing evaluation {eval_id}: {e}")
        meta_path = os.path.join(eval_dir, "eval_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["status"] = "Error"
                meta["error_message"] = str(e)
                meta["current_step"] = f"Error: {str(e)}"
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
        "status": "In Progress",
        "created_at": time.time()
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
                        meta = json.load(f)
                    if "created_at" not in meta:
                        meta["created_at"] = os.path.getctime(meta_path)
                    evals.append(meta)
                except Exception:
                    pass
    
    evals.sort(key=lambda x: x.get("created_at", 0), reverse=True)
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
                    logs = []
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        
                        if meta.get("status") == "Completed":
                            status_container.success("Answer generated successfully!")
                            break
                        elif meta.get("status") == "Error":
                            status_container.error(f"Error during generation: {meta.get('error_message', 'Unknown')}")
                            break
                            
                        logs = meta.get("logs", [])
                    
                    dots = (dots + 1) % 4
                    if logs:
                        log_display = "\n".join(logs) + f"\n...processing{'.' * dots}"
                        status_container.code(log_display, language="plaintext")
                    else:
                        status_container.info(f"Task queued and processing in background{'.' * dots}")
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
        from datetime import datetime
        
        for ev in evals:
            status = ev.get('status', 'Completed')
            status_icon = "⏳" if status == "In Progress" else "✅" if status == "Completed" else "❌"
            
            with st.container(border=True):
                dt = datetime.fromtimestamp(ev.get('created_at', 0))
                date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                doc_names = ", ".join([d['filename'] for d in ev['docs']])
                
                # Folosim un singur bloc markdown/HTML cu un line-height redus pentru a lipi elementele intre ele
                card_html = f"""
                <div style="line-height: 1.4; margin-bottom: -10px;">
                    <div style="font-weight: bold; margin-bottom: 4px;">{status_icon} Q: {ev['prompt'][:80]}...</div>
                    <div style="color: grey; font-size: 0.85em; margin-bottom: 2px;">📅 Query Date: {date_str}</div>
                    <div style="font-size: 0.95em; margin-bottom: 8px;">📄 <b>Context:</b> {doc_names}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                with st.expander("View detailed answer"):
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
