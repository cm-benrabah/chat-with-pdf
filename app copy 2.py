import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# UI setup
st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📄 Chat with Your PDF using DeepSeek (CPU-only)")

# PDF Upload
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
if pdf_file:
    with open("your.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load PDF
    loader = PyPDFLoader("your.pdf")
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Embed chunks
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embedding)

    # Load LLM from Ollama
    llm = Ollama(model="deepseek-r1:8b")

    # Setup QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    # CLI loop
    print("PDF loaded. Ask your questions (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke({"query": query})
        print("💬", response["result"])

        --------------------------------------

        import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# UI setup
st.set_page_config(page_title="PDF Chat", layout="wide")
st.title("📄 Chat with Your PDF using DeepSeek (CPU-only)")

# ✅ Cache the embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# PDF Upload
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
if pdf_file:
    with open("your.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load PDF
    loader = PyPDFLoader("your.pdf")
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # ✅ Use cached embedding model
    embedding = load_embedding_model()
    vectordb = FAISS.from_documents(docs, embedding)

    # Load LLM from Ollama
    llm = Ollama(model="deepseek-r1:8b")

    # Setup QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    # CLI loop
    print("PDF loaded. Ask your questions (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke({"query": query})
        print("💬", response["result"])
