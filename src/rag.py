"""
RAG From Scratch — a teaching implementation of Retrieval-Augmented Generation.

This builds a complete RAG pipeline with NO heavy frameworks, so every step is
visible and understandable:

    1. LOAD      - read source documents
    2. CHUNK     - split them into overlapping passages
    3. EMBED     - turn each chunk into a vector (via local Ollama)
    4. STORE     - keep vectors in a simple in-memory index (saved to JSON)
    5. RETRIEVE  - embed the question, find the most similar chunks (cosine sim)
    6. AUGMENT   - stuff those chunks into the prompt as context
    7. GENERATE  - ask the LLM to answer USING ONLY that context

Everything runs locally on Ollama — no API keys, no cloud cost.

Swap-in points for a "real" system are marked with:  # UPGRADE:
(e.g. replace the JSON store with Chroma/FAISS, or Ollama with a cloud API).
"""

import json
import math
import os
import pathlib
import urllib.request

# ----------------------------------------------------------------------------
# Config — override via environment variables.
# ----------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")  # embedding model
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2:1b")         # generation model
INDEX_PATH = pathlib.Path(__file__).parent.parent / "index.json"


# ----------------------------------------------------------------------------
# Step 3 helpers: talk to Ollama for embeddings and generation.
# ----------------------------------------------------------------------------
def _post(path, payload, timeout=120):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def embed(text):
    """Turn a piece of text into a vector (list of floats).

    An embedding places text in a high-dimensional space where semantically
    similar text ends up close together. This is the heart of retrieval.
    """
    out = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    return out["embedding"]


def generate(prompt):
    """Ask the chat model for a completion (non-streaming for simplicity)."""
    # Longer timeout: small local models can be slow to load on first call.
    out = _post("/api/generate", {"model": CHAT_MODEL, "prompt": prompt, "stream": False}, timeout=300)
    return out.get("response", "").strip()


# ----------------------------------------------------------------------------
# Step 2: chunking — split long docs into overlapping passages.
# Why overlap? So a sentence that straddles a boundary still appears whole in
# at least one chunk, which improves retrieval quality.
# ----------------------------------------------------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into ~chunk_size-character windows that overlap by `overlap`."""
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, leaving an overlap
    return [c for c in chunks if c.strip()]


# ----------------------------------------------------------------------------
# Step 5: similarity — cosine similarity between two vectors.
# Cosine sim measures the ANGLE between vectors (direction), not magnitude,
# which is what we want for comparing meaning.
# ----------------------------------------------------------------------------
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------------
# Steps 1-4: build the index from a folder of .txt/.md files.
# ----------------------------------------------------------------------------
def build_index(docs_dir):
    docs_dir = pathlib.Path(docs_dir)
    index = []  # list of {source, chunk_id, text, embedding}
    files = sorted(list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md")))
    if not files:
        raise SystemExit(f"No .txt or .md files found in {docs_dir}")

    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"  {f.name}: {len(chunks)} chunks")
        for i, ch in enumerate(chunks):
            index.append({
                "source": f.name,
                "chunk_id": i,
                "text": ch,
                "embedding": embed(ch),   # UPGRADE: batch this for speed at scale
            })

    INDEX_PATH.write_text(json.dumps(index))
    print(f"Indexed {len(index)} chunks from {len(files)} files -> {INDEX_PATH}")
    return index


def load_index():
    if not INDEX_PATH.exists():
        raise SystemExit("No index found. Run:  python3 src/rag.py index")
    return json.loads(INDEX_PATH.read_text())


# ----------------------------------------------------------------------------
# Step 5: retrieve the top-k most relevant chunks for a question.
# ----------------------------------------------------------------------------
def retrieve(question, index, k=3):
    q_vec = embed(question)
    scored = []
    for item in index:
        score = cosine_similarity(q_vec, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# ----------------------------------------------------------------------------
# Steps 6-7: augment the prompt with retrieved context, then generate.
# The instruction to answer ONLY from context is what reduces hallucination.
# ----------------------------------------------------------------------------
def answer(question, k=3):
    index = load_index()
    hits = retrieve(question, index, k=k)

    context_blocks = []
    for score, item in hits:
        context_blocks.append(f"[Source: {item['source']} #chunk {item['chunk_id']} | similarity {score:.3f}]\n{item['text']}")
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT "
        "below. If the answer is not in the context, say \"I don't know based on "
        "the provided documents.\" Cite the sources you used.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    response = generate(prompt)
    return response, hits


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    import sys
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print("\nUsage:\n  python3 src/rag.py index [docs_dir]\n  python3 src/rag.py ask \"your question\"")
        return

    cmd = args[0]
    if cmd == "index":
        docs_dir = args[1] if len(args) > 1 else (pathlib.Path(__file__).parent.parent / "sample_docs")
        print("Building index...")
        build_index(docs_dir)
    elif cmd == "ask":
        if len(args) < 2:
            print('Usage: python3 src/rag.py ask "your question"')
            return
        question = args[1]
        resp, hits = answer(question)
        print("\n=== RETRIEVED CONTEXT (what the model was given) ===")
        for score, item in hits:
            print(f"  [{score:.3f}] {item['source']} #chunk {item['chunk_id']}: {item['text'][:80]}...")
        print("\n=== ANSWER ===")
        print(resp)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
