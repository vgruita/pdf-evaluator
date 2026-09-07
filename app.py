import streamlit as st
import fitz  # PyMuPDF
import ollama
import base64
import gc

st.set_page_config(page_title="PDF Evaluator", layout="wide")

PAGE_EVAL_MODEL = "qwen3-vl:32b"
SYNTHESIS_MODEL = "llama3.1"

def get_base64_image(page):
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("utf-8")

def evaluate_page(criteria, text, base64_image):
    messages = []
    system_instruction = "You are an expert evaluator. Output ONLY valid Markdown. If the user prompt requests tables or specific formatting, you MUST provide them using Markdown syntax. All output MUST be in English."
    prompt = f"Evaluate the following page text based strictly on the user prompt.\n\nUser Prompt:\n{criteria}\n\nPage Text:\n{text}"
    
    messages.append({'role': 'system', 'content': system_instruction})
    messages.append({
        'role': 'user',
        'content': prompt,
        'images': [base64_image]
    })

    response = ollama.chat(
        model=PAGE_EVAL_MODEL, 
        messages=messages,
        options={"temperature": 0.0, "num_predict": 8192}
    )
    if 'error' in response:
        raise Exception(response['error'])
    return response.get('message', {}).get('content', '')

def synthesize_results(criteria, aggregated_reports):
    system_instruction = "You are an expert evaluator. Output ONLY valid Markdown. Follow any formatting or table requests specified in the user prompt. All output MUST be in English."
    prompt = f"Below is a page-by-page evaluation of a document against the following User Prompt:\n{criteria}\n\nPage Reports:\n{aggregated_reports}\n\nPlease generate a comprehensive 'Executive Summary' of the paper's compliance based strictly on the User Prompt."
    
    response = ollama.chat(
        model=SYNTHESIS_MODEL, 
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': prompt}
        ],
        options={"temperature": 0.0, "num_predict": 8192}
    )
    if 'error' in response:
        raise Exception(response['error'])
    return response.get('message', {}).get('content', '')

def main():
    st.title("Scientific PDF Evaluator")
    

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    criteria = st.text_area("User Prompt (Task & Criteria)", height=150, placeholder="Enter the detailed task and criteria to evaluate the document against. You can ask for tables and specific Markdown formatting here...")
    
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
        
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Processing", disabled=st.session_state.is_processing)
    with col2:
        stop_button = st.button("Stop/Cancel Processing", disabled=not st.session_state.is_processing)
        
    if stop_button:
        st.session_state.is_processing = False
        st.rerun()

    if start_button and uploaded_file and criteria:
        st.session_state.is_processing = True
        st.rerun()
    elif start_button and (not uploaded_file or not criteria):
        st.warning("Please upload a PDF and enter a user prompt before starting.")

    if st.session_state.is_processing:
        try:
            pdf_bytes = uploaded_file.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(doc)
            
            progress_bar = st.progress(0)
            
            aggregated_reports = ""
            
            with st.status("Starting evaluation...", expanded=True) as status:
                for i in range(num_pages):
                    status.update(label=f"Evaluating page {i+1}/{num_pages}...", state="running")
                    
                    page = doc.load_page(i)
                    text = page.get_text()
                    
                    base64_image = get_base64_image(page)
                    report = evaluate_page(criteria, text, base64_image)
                    aggregated_reports += f"## Page {i+1}\n\n{report}\n\n"
                    
                    # Explicit garbage collection
                    if base64_image:
                        del base64_image
                    del page
                    del text
                    gc.collect()
                    
                    progress_bar.progress((i + 1) / num_pages)
                    
                doc.close()
                
                status.update(label="Generating Executive Summary...", state="running")
                executive_summary = synthesize_results(criteria, aggregated_reports)
                
                final_report = f"# Executive Summary\n\n{executive_summary}\n\n# Page-by-Page Reports\n\n{aggregated_reports}"
                
                st.session_state.is_processing = False
                status.update(label="Processing complete!", state="complete", expanded=False)
            
            st.markdown(final_report)
            
            st.download_button(
                label="Download Report as Markdown",
                data=final_report,
                file_name="evaluation_report.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.is_processing = False
            
if __name__ == "__main__":
    main()
