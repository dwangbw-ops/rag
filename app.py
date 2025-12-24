import trafilatura
from langchain_core.documents import Document
import streamlit as st
import os
import pdfplumber
import re
from dotenv import load_dotenv
from openai import OpenAI

# --- RAG 核心库 ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    st.error("❌ 没找到密钥！请检查 .env 文件。")
    st.stop()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

st.set_page_config(page_title="Simplify - RAG", page_icon="🧠")
st.title("🧠 你的第二大脑 (PDF + URL)")

# --- 2. 核心函数 ---

def get_embedding_model():
    # 多语言模型，支持中英文混搜
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def clean_text(text):
    # 清洗工：去除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 新增功能: 网页抓取 ---
def get_web_content(url):
    """抓取 URL 正文"""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        return None

# --- 新增功能: 处理 URL 到向量库 ---
def process_url_to_vector_db(url):
    # 1. 抓取
    text_content = get_web_content(url)
    if not text_content:
        return None, "⚠️ 网页抓取失败（可能是反爬虫或链接无效）"
    
    # 2. 清洗
    text_content = clean_text(text_content)
    
    # 3. 切片
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    docs = [Document(page_content=x, metadata={"source": url}) for x in text_splitter.split_text(text_content)]
    
    # 4. 向量化 + 入库
    embeddings = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=None
    )
    return vector_db, f"✅ 成功索引网页！处理了 {len(docs)} 个片段。"

# --- 原有功能: 处理 PDF 到向量库 ---
def process_pdf_to_vector_db(uploaded_file):
    text_content = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    text_content += page_text + "\n"
    except Exception as e:
        return None, f"❌ PDF 读取错误: {e}"
    
    if not text_content:
        return None, "⚠️ PDF 内容为空（可能是纯图片）"

    text_content = clean_text(text_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    docs = [Document(page_content=x, metadata={"source": uploaded_file.name}) for x in text_splitter.split_text(text_content)]
    
    embeddings = get_embedding_model()
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=None
    )
    
    return vector_db, f"✅ 成功索引 PDF！处理了 {len(docs)} 个片段。"

# --- 3. 初始化 Session ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "把 PDF 扔进来，或者贴个网址，我来读。"}]
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "current_source" not in st.session_state:
    st.session_state.current_source = None

# --- 4. 侧边栏 ---
with st.sidebar:
    st.header("📚 数据源")
    
    # 选项 A: PDF
    st.subheader("📄上传本地文档")
    uploaded_file = st.file_uploader("选择 PDF", type="pdf")
    
    # 选项 B: URL
    st.markdown("---")
    st.subheader("🌐 或者输入文章链接")
    web_url = st.text_input("输入 URL (例如: 36kr/Bloomberg)")
    url_btn = st.button("抓取网页")

    # 逻辑路由
    if uploaded_file and uploaded_file.name != st.session_state.current_source:
        with st.spinner("正在处理 PDF..."):
            st.session_state.vector_db = None
            st.session_state.messages = []
            db, msg = process_pdf_to_vector_db(uploaded_file)
            if db:
                st.session_state.vector_db = db
                st.session_state.current_source = uploaded_file.name
                st.success(msg)
            else:
                st.error(msg)
    
    if url_btn and web_url:
        if web_url != st.session_state.current_source:
            with st.spinner(f"正在抓取 {web_url}..."):
                st.session_state.vector_db = None
                st.session_state.messages = []
                db, msg = process_url_to_vector_db(web_url)
                if db:
                    st.session_state.vector_db = db
                    st.session_state.current_source = web_url
                    st.success(msg)
                else:
                    st.error(msg)

    if st.button("🗑️ 清空所有"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.current_source = None
        st.rerun()

    st.markdown("---")
    show_debug = st.checkbox("🛠️ 开启调试模式")

# --- 5. 聊天界面 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("用中文问我..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- RAG 检索 ---
    context_text = ""
    if st.session_state.vector_db:
        results = st.session_state.vector_db.similarity_search(prompt, k=10)
        
        seen_content = set()
        unique_results = []
        for doc in results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)
            if len(unique_results) >= 4: 
                break
        
        for i, doc in enumerate(unique_results):
            # 获取来源 (是 PDF 名字 还是 URL)
            source = doc.metadata.get("source", "未知来源")
            context_text += f"\n[来源: {source} | 片段 {i+1}]: {doc.page_content}\n"
        
        if show_debug:
            with st.sidebar:
                st.subheader("🔍 AI 参考的片段")
                st.code(context_text, language="text")
    
    if context_text:
        full_prompt = f"你是一个学术助手。请根据以下参考片段回答用户问题。如果片段里没有答案，请直接说不知道。\n\n参考片段：\n{context_text}\n\n用户问题：{prompt}"
    else:
        full_prompt = prompt

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的学术助手。请用中文回答。"},
                {"role": "user", "content": full_prompt}
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})