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
from langchain_core.documents import Document

# 1. 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    st.error("❌ 没找到密钥！请检查 .env 文件。")
    st.stop()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

st.title("🧠 你的第二大脑 (学术论文专用版)")

# --- 2. 核心函数 ---
@st.cache_resource
def get_embedding_model():
    # ⚠️ 关键升级：换成支持中英文互搜的“多语言模型”
    # 以前那个只能搜英文，这个支持 50+ 种语言
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def clean_text(text):
    # 🧹 清洗工：把连在一起的单词强行拆开 (简单处理)
    # 并去除多余的换行符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_pdf_to_vector_db(uploaded_file):
    text_content = ""
    
    # A. 读取文字 (尝试更稳健的读取方式)
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            # layout=True 能保留文字的空间位置，减少单词粘连
            page_text = page.extract_text(layout=True)
            if page_text:
                text_content += page_text + "\n"
    
    if not text_content:
        return None, "⚠️ PDF 内容为空（可能是纯图片）"

    # B. 清洗数据
    text_content = clean_text(text_content)

    # C. 切豆腐 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] # 优先按句号切
    )
    docs = [Document(page_content=x) for x in text_splitter.split_text(text_content)]
    
    # D. 向量化 + 入库
    embeddings = get_embedding_model()
    
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory=None
    )
    
    return vector_db, f"✅ 成功索引！处理了 {len(docs)} 个片段。"

# --- 3. 初始化 Session ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "论文传上来，我来帮你读。"}]
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# --- 4. 侧边栏 ---
with st.sidebar:
    st.header("📂 文档上传")
    uploaded_file = st.file_uploader("上传 PDF", type="pdf")
    
    if uploaded_file:
        if uploaded_file.name != st.session_state.current_file_name:
            with st.spinner("检测到新论文，正在切换大脑 (首次下载多语言模型需 1 分钟)..."):
                st.session_state.vector_db = None
                st.session_state.messages = [] # 清空对话
                
                db, msg = process_pdf_to_vector_db(uploaded_file)
                if db:
                    st.session_state.vector_db = db
                    st.session_state.current_file_name = uploaded_file.name
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.info(f"当前文档：{uploaded_file.name}")

    if st.button("🗑️ 清空所有"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.session_state.current_file_name = None
        st.rerun()

# --- 5. 聊天界面 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 放在侧边栏的底部，作为高级选项
with st.sidebar:
    st.markdown("---")
    # ✅ 改动点：默认关闭调试模式，界面更干净
    show_debug = st.checkbox("🛠️ 开启调试模式 (Debug Mode)")

if prompt := st.chat_input("用中文问我关于论文的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- RAG 检索 ---
    context_text = ""
    if st.session_state.vector_db:
        # 搜索逻辑不变
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
            context_text += f"\n[参考片段 {i+1}]: {doc.page_content}\n"
        
        # ✅ 改动点：调试信息只显示在侧边栏，不干扰主对话
        if show_debug:
            with st.sidebar:
                st.subheader("🔍 AI 参考的片段")
                st.code(context_text, language="text") # 用代码块显示，更紧凑
    
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
