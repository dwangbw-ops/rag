__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import fitz  # PyMuPDF
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Simplify: Academic Copilot", layout="wide")

# --- UI Header ---
st.title("Simplify: Your Academic Copilot 🎓")
st.markdown("Powered by **DeepSeek V3** | Built for dual-column papers")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Select PDF", type="pdf")
    
    st.markdown("---")
    st.header("Or Paste URL")
    url_input = st.text_input("Article URL")
    fetch_btn = st.button("Fetch URL")
    
    st.markdown("---")
    st.markdown("### Settings")
    api_key = st.text_input("DeepSeek API Key (Optional)", type="password")
    debug_mode = st.checkbox("Debug Mode")

# --- Logic: PDF Processing (PyMuPDF) ---
def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        # Use "blocks" to handle dual-column layout correctly
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by vertical, then horizontal
        for b in blocks:
            text += b[4] + "\n"
    return text

# --- Main Logic ---
if uploaded_file or (url_input and fetch_btn):
    content = ""
    
    if uploaded_file:
        with st.spinner("Reading PDF (Dual-column mode)..."):
            try:
                content = process_pdf(uploaded_file)
                st.success("PDF processed successfully!")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
    
    elif url_input and fetch_btn:
        with st.spinner("Fetching URL..."):
            try:
                loader = WebBaseLoader(url_input)
                docs = loader.load()
                content = docs[0].page_content
                st.success("URL fetched successfully!")
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

    if content:
        # RAG Pipeline Setup
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(content)
        
        # Use DeepSeek API
        active_api_key = api_key if api_key else os.getenv("DEEPSEEK_API_KEY")
        
        if not active_api_key:
            st.warning("Please provide a DeepSeek API Key in the sidebar or .env file.")
        else:
            # Embedding & Vector Store
            # Note: Using OpenAIEmbeddings with DeepSeek base URL if compatible, 
            # or standard OpenAI embeddings if you have that key. 
            # For simplicity here, we assume standard setup or user has env vars set.
            # If using DeepSeek for everything, you might need a specific embedding setup.
            # Here we use a generic placeholder for the LLM part:
            
            try:
                # LLM Setup
                llm = ChatOpenAI(
                    model_name="deepseek-chat",
                    openai_api_key=active_api_key,
                    openai_api_base="https://api.deepseek.com/v1",
                    temperature=0.3
                )
                
                # Simple Chat Interface
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask me anything about the document..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):