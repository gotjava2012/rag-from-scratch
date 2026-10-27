# Vector Search and Vector Databases

Vector search finds items whose embedding vectors are closest to a query vector.
It powers semantic search, recommendation, and the retrieval step in RAG.

## Cosine similarity

Cosine similarity scores two vectors from -1 to 1 based on the angle between
them. A score near 1 means the texts are very similar in meaning, near 0 means
unrelated, and negative means opposite. Because it ignores vector magnitude, it
is robust to differences in text length.

## Exact vs approximate search

The simplest approach compares the query against every stored vector. This is
"exact" and easy to understand but slow when there are millions of vectors.
Production systems use Approximate Nearest Neighbor (ANN) algorithms such as
HNSW to trade a tiny amount of accuracy for a large speedup.

## Vector databases

A vector database stores embeddings and runs similarity search efficiently.
Popular choices include FAISS (a library), Chroma, Qdrant, Weaviate, Pinecone,
and pgvector (a PostgreSQL extension). They add features like metadata filters,
persistence, and horizontal scaling that a hand-written index lacks.

## When to use what

For learning and small datasets, an in-memory index with exact cosine
similarity is perfect — it makes the mechanics obvious. For production, move to
a dedicated vector database so you get speed, persistence, and filtering.
