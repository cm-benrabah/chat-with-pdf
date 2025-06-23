import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer

# ——— Embedding model loader ———
def get_embedding_model(path="./models/all-MiniLM-L6-v2"):
    if not os.path.exists(path):
        st.info("🔽 Downloading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(path)
    return HuggingFaceEmbeddings(model_name=path)

@st.cache_resource
def load_embedding_model():
    return get_embedding_model()

# ——— PDF loader & splitter ———
def load_pdf(file):
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# ——— Initialize app ———
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF")

# Upload
uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded:
    if "vectorstore" not in st.session_state:
        text = load_pdf(uploaded)
        st.success("PDF loaded.")
        chunks = split_text_into_chunks(text)
        st.info(f"{len(chunks)} chunks created.")
        emb = load_embedding_model()
        st.session_state.vectorstore = FAISS.from_texts(chunks, emb)
        st.session_state.chat_history = []
        st.session_state.query = ""  # holds current input

    # show history
    for idx, entry in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{idx}:** {entry['question']}")
        st.markdown(f"**A{idx}:** {entry['answer']}")

    # callback when hitting Enter
    def process_query():
        q = st.session_state.query.strip()
        if not q:
            return
        # show inline spinner
        placeholder = st.empty()
        placeholder.info("🤖 Thinking...")
        docs = st.session_state.vectorstore.as_retriever().get_relevant_documents(q)
        llm = ChatOllama(model="deepseek-r1:8b")
        qa = load_qa_chain(llm, chain_type="stuff")
        ans = qa.run(input_documents=docs, question=q)
        st.session_state.chat_history.append({"question": q, "answer": ans})
        st.session_state.query = ""         # clear input
        placeholder.empty()                # remove spinner

    # text_input with on_change; submit with Enter
    st.text_input("💬 Ask a question", key="query", on_change=process_query)

else:
    st.warning("Please upload a PDF to begin.")
