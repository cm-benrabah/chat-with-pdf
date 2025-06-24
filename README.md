# Chat with PDF using DeepSeek and LangChain

This Streamlit mini-project allows you to upload a PDF document and ask questions about its content. The application uses local embeddings and a DeepSeek model through LangChain and Ollama to provide accurate answers based on the document.

## 🚀 Features

- Upload and parse PDF documents
- Split content into semantic chunks
- Generate local embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- Store embeddings in a FAISS vector database
- Answer queries using a local LLM (ChatOllama).
- Preserve conversation history
- Fully offline: no OpenAI key required
- Use FAISS for similarity search.
- Fallback to raw LLM answer if question is not relevant to the document.
- Streamlit interface with chat history.
## 🛠️ Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed with the model `deepseek-r1:8b` pulled
- Dependencies from `requirements.txt` (see below)

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

2. (Optional) Create and activate a virtual environment:

```bash
conda create -n LangChain python=3.10 -y
conda activate LangChain
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Pull the DeepSeek model via Ollama:

```bash
ollama pull deepseek-r1:8b
```

5. Run the app:

```bash
streamlit run app.py --server.fileWatcherType none
```

## 📁 Directory Structure

```
chat-with-pdf/
├── app.py
├── models/                # Embedding model folder (empty in repo)
├── README.md
└── requirements.txt
```
## ⚙️ How It Works

1. PDF is uploaded and parsed to plain text.
2. Text is split into overlapping chunks.
3. Each chunk is embedded using a local embedding model (`all-MiniLM-L6-v2`).
4. When a user submits a question:
   - The similarity of the query is checked with FAISS.
   - If a relevant match is found, answer is generated via **RAG (Retrieve-then-Generate)**.
   - If no match is strong enough (above threshold), the LLM answers the prompt without retrieval.

---

## 📝 Customization & Notes

- The similarity threshold can be tuned:
---
```python
threshold = 1  # Lower value = more strict match
## 📝 Notes

- The `models/` folder is excluded from Git to avoid pushing large files. A `.gitkeep` or `.empty` file is used to keep it in the repo structure.
- You can switch LLMs by updating the `ChatOllama(model="deepseek-r1:8b")` line.

## 📄 License

MIT License
