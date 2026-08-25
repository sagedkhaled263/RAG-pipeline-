"""
Stage 3: Online Query-Time Pipeline (Interactive Mode)
Takes a live question from the user in the terminal, retrieves relevant
chunks, constructs context, and generates a grounded answer using Groq.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rerank import rerank

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print("Loading vector store...")
index = faiss.read_index(os.path.join(SCRIPT_DIR, "vector_store.faiss"))
with open(os.path.join(SCRIPT_DIR, "vector_store_meta.json"), "r") as f:
    meta = json.load(f)

embed_model = SentenceTransformer(MODEL_NAME)


def retrieve(query, top_k=TOP_K, country_filter=None):
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    fetch_n = top_k * 4 if country_filter else top_k
    scores, ids = index.search(q_vec, fetch_n)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = meta[idx]
        if country_filter and chunk["metadata"].get("country") != country_filter:
            continue
        results.append({**chunk, "score": float(score)})
        if len(results) >= top_k:
            break
    return results

def retrieve_multi(sub_queries, top_k_each=3):
    seen_ids = set()
    merged = []
    for sq in sub_queries:
        for r in retrieve(sq, top_k=top_k_each):
            if r["chunk_id"] not in seen_ids:
                seen_ids.add(r["chunk_id"])
                merged.append(r)
    return merged


def construct_context(results):
    seen = set()
    lines = []
    for i, r in enumerate(results, 1):
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        src = r["metadata"]["source_document"]
        sec = r["metadata"]["section"]
        lines.append(f"[Chunk {i} | Source: {src} | Section: {sec}]\n{r['text']}")
    return "\n\n".join(lines)


def build_prompt(query, context):
    return f"""You are a company knowledge assistant. Use ONLY the provided
context to answer the question. If the answer is not in the context, say
"I don't know based on the available documents." Cite the source document
for each claim.

Context:
{context}

User Question:
{query}

Answer:"""


def generate_answer(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def ask(query, country_filter=None):
    algo_keywords = {
        "k-means": "K-Means clustering algorithm",
        "kmeans": "K-Means clustering algorithm",
        "hierarchical": "Hierarchical clustering algorithm",
        "dbscan": "DBSCAN clustering algorithm",
        "gmm": "Gaussian Mixture Model clustering",
        "gaussian mixture": "Gaussian Mixture Model clustering",
    }
    q_lower = query.lower()
    matched = {v for k, v in algo_keywords.items() if k in q_lower}
    print(f"DEBUG - matched sub-queries: {matched}")   # ← add this line temporarily

    if len(matched) >= 2:
        results = retrieve_multi(list(matched), top_k_each=3)
        print("\n-- Multi-query retrieval used (multiple topics detected) --")
    else:
        results = retrieve(query, top_k=TOP_K, country_filter=country_filter)

    print("\n-- Retrieved chunks --")
    for r in results:
        print(f"  score={r['score']:.3f}  {r['metadata']['source_document']}"
              f"  |  {r['metadata']['section']}")

    context = construct_context(results)
    prompt = build_prompt(query, context)
    answer = generate_answer(prompt)

    if answer:
        print("\n-- Generated Answer (Groq) --")
        print(answer)
    else:
        print("\n-- No GROQ_API_KEY set. Paste this prompt into any chat: --")
        print(prompt)

if __name__ == "__main__":
    print("\nRAG Assistant ready. Type your question, or 'exit' to quit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        # Optional: simple auto-detect for UAE-specific questions
        country_filter = "UAE" if "uae" in query.lower() else None

        ask(query, country_filter=country_filter)
        print("\n" + "=" * 80 + "\n")