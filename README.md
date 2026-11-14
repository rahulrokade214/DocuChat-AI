# Groq RAG

Chat with your PDFs / text files. Zero compiled dependencies, so it runs on locked-down Windows PCs.

**Stack:** Python standard library HTTP server + BM25 retrieval + Groq Llama (REST) + browser UI. Only `pypdf` (pure Python) is installed.

## Run
```
python -m pip install --user pypdf
copy .env.example .env      (paste your Groq key into .env)
python server.py
```
Open http://localhost:8000, upload files, click **Index documents**, ask questions.

## Files
- `server.py` HTTP server + API routes
- `index.html` chat UI
- `rag/store.py` file loading, chunking, BM25 retrieval
- `rag/llm.py` Groq REST call (urllib)
