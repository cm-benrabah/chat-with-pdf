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

# ——— Initialize Streamlit App ———
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if "vectorstore" not in st.session_state:
        text = load_pdf(uploaded)
        st.success("✅ PDF loaded.")
        chunks = split_text_into_chunks(text)
        st.info(f"🔹 {len(chunks)} chunks created.")
        emb = load_embedding_model()
        st.session_state.vectorstore = FAISS.from_texts(chunks, emb)
        st.session_state.chat_history = []
        st.session_state.query = ""

    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")

    def process_query():
        query = st.session_state.query.strip()
        if not query:
            return

        placeholder = st.empty()
        placeholder.info("🤖 Generating answer...")

        retriever = st.session_state.vectorstore.as_retriever()
        results = st.session_state.vectorstore.similarity_search_with_score(query, k=3)

        # Show scores for transparency
        threshold = 1
        relevant_docs = []
        for doc, score in results:
            st.markdown(f"🔎 Score: `{score:.3f}` → {'✅ Relevant' if score < threshold else '❌ Irrelevant'}")
            if score < threshold:
                relevant_docs.append(doc)

        llm = ChatOllama(model="mistral")  # Replace with your Ollama model if needed

        if relevant_docs:
            qa = load_qa_chain(llm, chain_type="stuff")
            answer = qa.run(input_documents=relevant_docs, question=query)
            tag = "_🔍 Answered using RAG (relevant documents)_"
        else:
            answer = llm.invoke(query).content
            tag = "_💬 Answered using LLM only (no relevant content found)_"

        #answer = "empty answer"
        st.session_state.chat_history.append({
            "question": query,
            "answer": f"{answer}\n\n{tag}"
        })

        st.session_state.query = ""
        placeholder.empty()

    st.text_input("💬 Ask a question", key="query", on_change=process_query)

else:
    st.warning("⬆️ Please upload a PDF to get started.")
