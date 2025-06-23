import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer

# ========== Load or Download Embedding Model ==========
def get_embedding_model(path="./models/all-MiniLM-L6-v2"):
    if not os.path.exists(path):
        st.info("🔽 Downloading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(path)
    else:
        print("✅ Using cached model from:", path)
    return HuggingFaceEmbeddings(model_name=path)

@st.cache_resource
def load_embedding_model():
    return get_embedding_model()

# ========== Load and Split PDF ==========
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    raw_text = load_pdf(uploaded_file)
    st.success("✅ PDF loaded successfully.")

    chunks = split_text_into_chunks(raw_text)
    st.info(f"🔹 Document split into {len(chunks)} chunks.")

    embedding = load_embedding_model()

    # Generate embeddings and store in FAISS
    with st.spinner("🔍 Generating vector store..."):
        vectorstore = FAISS.from_texts(chunks, embedding)
    st.success("✅ Embeddings stored in vector DB (FAISS).")

    # Display sample chunks
    st.write("### Sample Chunks")
    for i, chunk in enumerate(chunks[:3]):
        st.text(f"Chunk {i+1}: {chunk[:300]}...")

    # Prompt input (multi-prompt supported)
    query = st.text_input("💬 Ask a question about the PDF")

    if query:
        with st.spinner("🤖 Generating answer from Deepseek..."):
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(query)

            llm = ChatOllama(model="deepseek-r1:8b")  # or deepseek-r1:8b if you prefer
            qa_chain = load_qa_chain(llm, chain_type="stuff")

            answer = qa_chain.run(input_documents=docs, question=query)

            # Store in history
            st.session_state.chat_history.append({"question": query, "answer": answer})

    # Display chat history
    if st.session_state.chat_history:
        st.write("### 🧠 Conversation History")
        for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
            st.markdown(f"**Q{i}:** {entry['question']}")
            st.markdown(f"**A{i}:** {entry['answer']}")

else:
    st.warning("⬆️ Please upload a PDF to get started.")
