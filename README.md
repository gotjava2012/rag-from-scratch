# RAG From Scratch

A complete **Retrieval-Augmented Generation (RAG)** pipeline built with no heavy
frameworks, so every step is visible: chunk → embed → store → retrieve → augment
→ generate. Runs fully **locally on Ollama** (no API keys, no cloud cost).

## What it demonstrates
- Document chunking with overlap
- Embeddings and **cosine-similarity** vector search (implemented by hand)
- Grounded generation that answers **only** from retrieved context (anti-hallucination)
- Transparent retrieval: shows exactly which chunks the model was given

## Quick start
```bash
# Prereqs: Python 3 + Ollama (https://ollama.com)
ollama pull nomic-embed-text
ollama pull llama3.2:1b        # or any small chat model

python3 src/rag.py index                    # build the vector index
python3 src/rag.py ask "What is cosine similarity and why is it used in RAG?"
```

## Learn it step by step
See **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — a full walkthrough of RAG concepts
with exercises and production-upgrade notes (vector DBs, re-ranking, hybrid search).

## Project layout
```
src/rag.py          # the RAG pipeline (heavily commented)
sample_docs/        # example documents to query
docs/TUTORIAL.md    # step-by-step tutorial
```

## Stack
Python (standard library only) · Ollama · nomic-embed-text · cosine similarity
