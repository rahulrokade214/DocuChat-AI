# 📚 DocuChat AI

A Retrieval-Augmented Generation (RAG) assistant that lets you chat with your own PDF, TXT and Markdown files. It is built with the **Groq API** and has **no compiled dependencies**, so it runs even on locked-down Windows machines where packages like `torch` or `faiss` are blocked.

## ✨ Features

- Upload PDF / TXT / MD files and ask questions about them
- Answers are grounded in your documents, with inline citations like `[file.pdf p.2]`
- "Sources" panel shows the exact text used for each answer
- Keyword retrieval with the BM25 ranking algorithm, written in pure Python
- Browser chat UI with light and dark mode
- Automatically switches to an available Groq model if the configured one is retired
- Index is saved to disk, so documents survive restarts

## 🧠 How it works

```
Upload file → extract text → split into overlapping chunks → BM25 index
Question → rank chunks → top 4 chunks + question → Groq LLM → answer + sources
```

## 🛠️ Tech stack

| Part | Technology |
|------|------------|
| LLM | Groq API (OpenAI-compatible REST endpoint) |
| Retrieval | BM25, pure Python |
| Backend | Python standard library (`http.server`, `urllib`) |
| PDF parsing | `pypdf` (pure Python) |
| Frontend | HTML, CSS and vanilla JavaScript |

## 📁 Project structure

```
groq-rag-lite/
├── server.py          # HTTP server and API routes
├── index.html         # Chat user interface
├── requirements.txt   # Only pypdf
├── .env.example       # Template for your API key
├── rag/
│   ├── store.py       # File loading, chunking, BM25 retrieval
│   └── llm.py         # Groq API call with model fallback
└── data/
    └── sample.txt     # Small file for testing
```

## 🚀 Getting started

### 1. Prerequisites
- Python 3.10 or newer
- A free Groq API key from https://console.groq.com/keys

### 2. Install

```bash
git clone https://github.com/rahulrokade214/groq-rag-lite.git
cd groq-rag-lite
python -m pip install -r requirements.txt
```

### 3. Add your API key

Copy `.env.example` to `.env` and paste your key:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

You can also paste the key into the sidebar box in the app instead.

### 4. Run

```bash
python server.py
```

Open http://localhost:8000, upload a document, click **Index documents**, and start asking questions.

## 💬 Example questions

- What skills are listed in this document?
- Summarize this document.
- Where did the person do their internship?

## ⚠️ Limitations

- Retrieval is keyword-based, so questions work best when they share words with the document
- Scanned PDFs (images without text) cannot be read
- The index is held in memory, so it suits small to medium document sets

## 🔮 Future improvements

- Embedding-based semantic search on machines that allow compiled packages
- Hybrid BM25 + vector retrieval and a reranking step
- DOCX and web page loaders
- Streaming answers
- A small evaluation set to measure answer accuracy

## 🔒 Security

Never commit your `.env` file. It is already listed in `.gitignore`. If a key is ever exposed, delete it in the Groq console and create a new one.

## 👤 Author

**Rahul Rokade**: Data Analyst and Agentic AI developer
GitHub: [rahulrokade214](https://github.com/rahulrokade214)
