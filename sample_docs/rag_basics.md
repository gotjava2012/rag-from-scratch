# Retrieval-Augmented Generation (RAG) — Overview

Retrieval-Augmented Generation (RAG) is a technique that improves the responses
of a large language model by giving it relevant external information at query
time, instead of relying only on what the model learned during training.

## Why RAG exists

Language models have two well-known limitations. First, their knowledge is
frozen at training time, so they do not know about recent or private
information. Second, they can "hallucinate" — produce confident but incorrect
answers. RAG addresses both by retrieving real source documents and instructing
the model to answer using only those documents.

## The core pipeline

A RAG system has two phases. The indexing phase runs once, ahead of time: source
documents are split into chunks, each chunk is converted into an embedding
vector, and the vectors are stored in a vector index. The query phase runs for
every question: the question is embedded, the most similar chunks are retrieved
from the index, those chunks are inserted into the prompt as context, and the
model generates an answer grounded in that context.

## Embeddings

An embedding is a list of numbers that represents the meaning of a piece of
text. Texts with similar meaning produce vectors that are close together in the
embedding space. Similarity is usually measured with cosine similarity, which
compares the angle between two vectors rather than their length.

## Chunking

Documents are split into smaller passages called chunks because models have a
limited context window and because smaller, focused passages retrieve more
precisely. Chunks often overlap slightly so that a sentence spanning a boundary
still appears intact in at least one chunk.
