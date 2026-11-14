"""Pure-Python RAG store: file loading, chunking and BM25 retrieval. No compiled libraries."""
import base64
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
STORAGE = Path("storage")

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
        "were", "it", "this", "that", "with", "as", "by", "at", "from", "be", "what",
        "how", "does", "do", "which", "who", "about"}


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_pages(name, raw):
    """Return [(page_number, text)] from file bytes."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader  # pure-Python
        reader = PdfReader(io.BytesIO(raw))
        return [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, start=1)]
    if ext in {".txt", ".md"}:
        return [(1, raw.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {ext or name}")


class Store:
    k1, b = 1.5, 0.75

    def __init__(self):
        self.docs = []
        f = STORAGE / "docs.json"
        if f.exists():
            self.docs = json.loads(f.read_text(encoding="utf-8"))
        self._build()

    def _build(self):
        toks = [tokenize(d["text"]) for d in self.docs]
        self.tf = [Counter(t) for t in toks]
        self.lens = [len(t) for t in toks]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 1
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def _save(self):
        STORAGE.mkdir(exist_ok=True)
        (STORAGE / "docs.json").write_text(json.dumps(self.docs), encoding="utf-8")

    def add_file(self, name, b64data):
        raw = base64.b64decode(b64data)
        new = []
        for page, text in read_pages(name, raw):
            for piece in chunk_text(text):
                new.append({"source": name, "page": page, "text": piece})
        # replace older copy of same file
        self.docs = [d for d in self.docs if d["source"] != name] + new
        self._build()
        self._save()
        return len(new)

    def search(self, query, k=4):
        q = tokenize(query)
        if not self.docs or not q:
            return []
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1)))
            scores.append(s)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.docs[i], "score": round(scores[i], 3)} for i in top if scores[i] > 0]

    def sources(self):
        return sorted({d["source"] for d in self.docs})

    def clear(self):
        self.docs = []
        self._build()
        (STORAGE / "docs.json").unlink(missing_ok=True)
