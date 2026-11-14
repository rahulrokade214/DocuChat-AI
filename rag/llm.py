"""Call the Groq REST API with only the standard library."""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
- If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
- Be concise and accurate. Do not invent facts.
- Cite sources inline like [source.pdf p.3]."""


def load_env(path=".env"):
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def answer(api_key, model, question, hits, history):
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits) \
        or "No relevant text was found in the uploaded documents."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m for m in history[-6:] if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    body = json.dumps({"model": model, "messages": messages, "temperature": 0.2}).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "groq-rag-lite/1.0",  # Groq's firewall rejects Python's default agent
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(f"Groq API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")
