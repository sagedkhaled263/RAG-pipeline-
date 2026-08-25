"""
Stage 4: Evaluation — Retrieval Recall & Precision
Compares what the pipeline actually retrieves against a manually
labeled ground-truth set of relevant chunk_ids per question.

Recall  = relevant chunks retrieved / total relevant chunks that exist
Precision = relevant chunks retrieved / total chunks retrieved
"""
from statistics import mean
from rerank import rerank

# Reuse everything already built in 03_query.py (embedding model, FAISS
# index, retrieve(), retrieve_multi()) instead of duplicating the pipeline.
from importlib import import_module
import sys
sys.modules.setdefault("query_module", None)

# 03_query.py has an interactive input() loop guarded by
# `if __name__ == "__main__":`, so importing it is safe — none of that
# loop runs, only the top-level setup (model/index loading) and function
# definitions execute.
import importlib.util
spec = importlib.util.spec_from_file_location("query_module", "03_query.py")
query_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_module)

retrieve = query_module.retrieve
retrieve_multi = query_module.retrieve_multi
TOP_K = query_module.TOP_K


# ---- Ground truth: manually judged relevant chunk_ids per question ----
EVAL_SET = [
    {
        "question": "What is the onboarding process for a newly hired employee in Canada?",
        "relevant_ids": {"HR_Policy_Canada.docx_001"},
        "country_filter": None,
        "sub_queries": None,
    },
    {
        "question": "How can an employee request annual leave or report an absence?",
        "relevant_ids": {
            "HR_Policy_Canada.docx_004", "HR_Policy_Canada.docx_005",
            "HR_Policy_Egypt.docx_004", "HR_Policy_Egypt.docx_005",
            "HR_Policy_United_Arab_Emirates.docx_004", "HR_Policy_United_Arab_Emirates.docx_005",
            "HR_Policy_United_States.docx_004", "HR_Policy_United_States.docx_005",
        },
        "country_filter": None,
        "sub_queries": None,
    },
    {
        "question": "How does the company handle workplace misconduct and disciplinary actions in UAE?",
        "relevant_ids": {"HR_Policy_United_Arab_Emirates.docx_008"},
        "country_filter": "UAE",
        "sub_queries": None,
    },
    {
        "question": "How are payroll, compensation, and employee benefits administered?",
        "relevant_ids": {
            "HR_Policy_Canada.docx_006", "HR_Policy_Egypt.docx_006",
            "HR_Policy_United_Arab_Emirates.docx_006", "HR_Policy_United_States.docx_006",
        },
        "country_filter": None,
        "sub_queries": None,
    },
    {
        "question": "How does a neural network learn the correct weights, and what roles do forward propagation, loss calculation, backpropagation, and gradient descent play in the training process?",
        "relevant_ids": {"ML.docx_012", "ML.docx_013", "ML.docx_016", "ML.docx_017", "ML.docx_020", "ML.docx_029"},
        "country_filter": None,
        "sub_queries": None,
    },
    {
        "question": "What are the main differences between K-Means, Hierarchical Clustering, DBSCAN, and GMM, and when would you choose each algorithm?",
        "relevant_ids": {"ML.docx_035", "ML.docx_046", "ML.docx_048", "ML.docx_053", "ML.docx_071"},
        "country_filter": None,
        "sub_queries": [
            "K-Means clustering algorithm",
            "Hierarchical clustering algorithm",
            "DBSCAN clustering algorithm",
            "Gaussian Mixture Model clustering",
        ],
    },
]


def evaluate_one(item, retrieve_k=15, final_k=5):
    if item["sub_queries"]:
        candidates = retrieve_multi(item["sub_queries"], top_k_each=6)
    else:
        candidates = retrieve(item["question"], top_k=retrieve_k, country_filter=item["country_filter"])

    results = rerank(item["question"], candidates, top_k=final_k)

    retrieved_ids = {r["chunk_id"] for r in results}
    relevant_ids = item["relevant_ids"]

    true_positives = retrieved_ids & relevant_ids
    precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(true_positives) / len(relevant_ids) if relevant_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "question": item["question"],
        "retrieved": retrieved_ids,
        "relevant": relevant_ids,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    all_results = []
    for i, item in enumerate(EVAL_SET, 1):
        res = evaluate_one(item)
        all_results.append(res)

        print("=" * 80)
        print(f"Q{i}: {res['question']}")
        print("=" * 80)
        print(f"  Relevant (ground truth):  {sorted(res['relevant'])}")
        print(f"  Retrieved:                {sorted(res['retrieved'])}")
        print(f"  Matched (true positives):  {sorted(res['true_positives'])}")
        print(f"  Precision: {res['precision']:.2f}   Recall: {res['recall']:.2f}   F1: {res['f1']:.2f}")
        print()

    print("=" * 80)
    print("OVERALL AVERAGES")
    print("=" * 80)
    print(f"  Average Precision: {mean(r['precision'] for r in all_results):.3f}")
    print(f"  Average Recall:    {mean(r['recall'] for r in all_results):.3f}")
    print(f"  Average F1:        {mean(r['f1'] for r in all_results):.3f}")