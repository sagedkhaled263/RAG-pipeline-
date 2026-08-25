"""
Stage 4: Reranking
Takes a wide set of candidate chunks from the retriever (high recall,
so-so precision) and reranks them using a cross-encoder, which jointly
scores the query and each chunk together, then keeps only the best few
(higher precision).
"""
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Loading cross-encoder reranker...")
reranker = CrossEncoder(RERANKER_MODEL)


def rerank(query, candidates, top_k=5):
    """
    Jointly score (query, chunk_text) pairs with a cross-encoder and
    return only the top_k highest-scoring chunks.
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    test_query = "How do I request annual leave?"
    test_candidates = [
        {"text": "Employees should submit leave requests through the approved HR system in advance.", "chunk_id": "A"},
        {"text": "The company was founded to provide consulting services across multiple regions.", "chunk_id": "B"},
        {"text": "Annual leave balances are tracked and reviewed by HR each quarter.", "chunk_id": "C"},
    ]
    top = rerank(test_query, test_candidates, top_k=2)
    print("\nTest reranking result:")
    for r in top:
        print(f"  score={r['rerank_score']:.3f}  {r['chunk_id']}  |  {r['text'][:60]}")