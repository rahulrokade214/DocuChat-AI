"""Call the Groq REST API with only the standard library.
If the configured model is unavailable, the app asks Groq which models your key can use and picks one."""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS_URL = "https://api.groq.com/openai/v1/models"

PREFERRED = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.1-8b-instant",
             "llama-3.3-70b-versatile", "qwen/qwen3-32b"]
SKIP = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "compound")
_resolved = {}  # requested model -> model that actually works

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


def _call(url, api_key, body=None):
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "groq-rag-lite/1.0",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def pick_model(api_key):
    ids = [m["id"] for m in _call(MODELS_URL, api_key)["data"]]
    for p in PREFERRED:
        if p in ids:
            return p
    for i in ids:
        if not any(s in i.lower() for s in SKIP):
            return i
    raise RuntimeError("No chat models are available for this API key.")


def answer(api_key, model, question, hits, history):
    context = "\n\n".join(f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits) \
        or "No relevant text was found in the uploaded documents."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m for m in history[-6:] if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    def chat(m):
        body = json.dumps({"model": m, "messages": messages, "temperature": 0.2}).encode()
        return _call(CHAT_URL, api_key, body)["choices"][0]["message"]["content"]

    try:
        model = _resolved.get(model, model)
        try:
            return chat(model)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 404):
                raise
            fallback = pick_model(api_key)
            if fallback == model:
                raise
            _resolved[model] = fallback
            print(f"Model '{model}' unavailable, using '{fallback}' instead.")
            return chat(fallback)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(f"Groq API error {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")