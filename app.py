__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import fitz  # PyMuPDF
import os
from langchain_community.document_loaders import WebBaseLoader
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
content = ""

# 1. Handle Inputs
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

# 2. Chat Interface
if content:
    # Use DeepSeek API
    active_api_key = api_key if api_key else os.getenv("DEEPSEEK_API_KEY")
    
    if not active_api_key:
        st.warning("Please provide a DeepSeek API Key in the sidebar or .env file to start chatting.")
    else:
        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle User Input
        if prompt := st.chat_input("Ask me anything about the document..."):
            # Add user message to state
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI Response
            with st.chat_message("assistant"):
                try:
                    # Setup LLM
                    llm = ChatOpenAI(
                        model_name="deepseek-chat",
                        openai_api_key=active_api_key,
                        openai_api_base="https://api.deepseek.com",
                        temperature=0.3
                    )
                    
                    # Context Injection (Simplified for stability)
                    # We limit context to first 25k chars to avoid token limits in this demo
                    truncated_content = content[:25000] 
                    
                    system_msg = f"""You are a helpful academic assistant. 
                    Answer the user's question based strictly on the context below.
                    If the answer is not in the context, say you don't know.
                    
                    Context:
                    {truncated_content}
                    """
                    
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = llm.invoke(messages)
                    st.markdown(response.content)
                    
                    # Add AI response to state
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                except Exception as e:
                    st.error(f"API Error: {e}")