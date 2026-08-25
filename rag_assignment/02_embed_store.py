"""
Stage 2: Embedding generation + Vector DB storage
Offline pipeline steps 4-5 from the lecture.
"""
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# A small, fast, good-quality local embedding model (384 dimensions).
# Swap to "all-mpnet-base-v2" later for higher quality if you want.
MODEL_NAME = "all-MiniLM-L6-v2"

with open("/home/mohamedehab/ai_itida/rag_assignment/chunks.json", "r") as f:
    chunks = json.load(f)

print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

texts = [c["text"] for c in chunks]
print(f"Embedding {len(texts)} chunks...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype="float32")

# normalize_embeddings=True + IndexFlatIP == cosine similarity search
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "vector_store.faiss")

# Vector DB "payload": chunk text + metadata, aligned by row order with the index
with open("vector_store_meta.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Vector store built: {index.ntotal} vectors, dim={dim}")