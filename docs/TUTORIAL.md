# Tutorial: Build RAG From Scratch (Step by Step)

This tutorial teaches **Retrieval-Augmented Generation (RAG)** by building a
complete, working pipeline with no heavy frameworks — so you can see every
moving part. By the end you will understand embeddings, chunking, vector
similarity, retrieval, and grounded generation, and you will have a tool that
answers questions about your own documents.

Everything runs **locally on Ollama** — no API keys, no cloud cost.

---

## What is RAG, in one paragraph

A language model only knows what it learned during training, and it sometimes
makes things up. RAG fixes both problems: before answering, it **retrieves**
relevant passages from your documents and tells the model to answer **using only
those passages**. The model's broad language ability is combined with your
specific, up-to-date facts.

---

## Prerequisites

1. **Python 3** (check: `python3 --version`)
2. **Ollama** installed and running — https://ollama.com
3. Pull the two models this project uses:
   ```bash
   ollama pull nomic-embed-text    # turns text into embedding vectors
   ollama pull llama3.2:3b         # generates the final answer
   ```
   (On a small machine, any small chat model works — set `CHAT_MODEL` to one you have.)
4. Install the one Python dependency:
   ```bash
   pip3 install requests   # only used indirectly; the code uses urllib, so this is optional
   ```
   (The code uses only the Python standard library, so there is nothing else to install.)

---

## Step 1 — Load your documents

Put `.txt` or `.md` files in `sample_docs/`. Two example files are included
(`rag_basics.md`, `vector_search.md`). Later, drop in your own notes, docs, or
any text you want to ask questions about.

## Step 2 — Chunking (why we split documents)

Models have a limited **context window**, and smaller passages retrieve more
precisely. So we split each document into ~500-character **chunks** that
**overlap** by 100 characters. The overlap means a sentence sitting on a chunk
boundary still appears whole in at least one chunk.

See `chunk_text()` in `src/rag.py`.

## Step 3 — Embeddings (turning text into vectors)

An **embedding** is a list of numbers that captures the *meaning* of text.
Similar meanings produce vectors that are close together. We get embeddings
from Ollama's `nomic-embed-text` model. See `embed()`.

## Step 4 — Build the index

Run:
```bash
python3 src/rag.py index
```
This chunks every document, embeds each chunk, and saves everything to
`index.json`. You will see how many chunks each file produced. The indexing
phase runs **once** (re-run it when your documents change).

## Step 5 — Retrieval (find the most relevant chunks)

When you ask a question, we embed the **question** the same way, then compare it
to every stored chunk using **cosine similarity** (the angle between vectors —
closer angle = more similar meaning). We keep the top `k` chunks. See
`cosine_similarity()` and `retrieve()`.

## Step 6 — Augment the prompt

We insert the retrieved chunks into the prompt as **CONTEXT**, and instruct the
model: *answer using ONLY this context; if it's not here, say you don't know;
cite your sources.* This instruction is what dramatically reduces
hallucination. See the prompt in `answer()`.

## Step 7 — Generate the answer

Run:
```bash
python3 src/rag.py ask "What is cosine similarity and why is it used in RAG?"
```
You will see two things:
1. **The retrieved context** — exactly what the model was given (transparency!).
2. **The answer** — grounded in your documents, with sources cited.

Try a question your documents can't answer, e.g.:
```bash
python3 src/rag.py ask "What is the capital of France?"
```
A good RAG system should say it doesn't know *based on the provided documents* —
that's the anti-hallucination behavior working.

---

## Exercises (to deepen your understanding)

1. **Change the chunk size** (e.g. 200 vs 1000) and re-index. How do the
   retrieved chunks and answers change?
2. **Change `k`** (number of retrieved chunks) in `retrieve()`. More context
   isn't always better — why?
3. **Add your own documents** to `sample_docs/` and ask about them.
4. **Inspect `index.json`** — look at an embedding. It's just a list of floats.

---

## How to make this production-grade (next steps)

This project is intentionally simple so the concepts are visible. In the code,
every place marked `# UPGRADE:` is where a real system would differ:

- **Vector store:** replace the JSON file + exact cosine loop with a vector DB
  (Chroma, FAISS, Qdrant, or pgvector) for speed, persistence, and metadata
  filtering. Exact search is O(n); production uses Approximate Nearest Neighbor.
- **Batch embedding:** embed chunks in batches instead of one at a time.
- **Better chunking:** split on sentence/paragraph boundaries or by tokens.
- **Re-ranking:** retrieve more chunks, then re-rank with a cross-encoder.
- **Hybrid search:** combine vector similarity with keyword (BM25) search.
- **Evaluation:** measure retrieval quality (hit rate, MRR) and answer quality.
- **Models:** swap Ollama for a cloud API by changing `embed()`/`generate()`.

Understanding the simple version first makes all of these upgrades make sense.
