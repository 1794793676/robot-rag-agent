"""Offline retrieval evaluation and threshold calibration."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    cases: list[dict[str, Any]] = []
    offset = 0
    for group in payload["groups"]:
        for index in range(group["count"]):
            number = offset + index
            kind = group["kind"]
            relevant_id = f"relevant-{number}"
            documents = [{
                "doc_id": f"distractor-{number}",
                "database_id": "other" if kind == "cross_database" else "primary",
                "text": group["distractor_template"].format(i=index),
            }]
            relevant_ids: list[str] = []
            if kind == "positive":
                relevant_ids.append(relevant_id)
                documents.append({
                    "doc_id": relevant_id,
                    "database_id": "primary",
                    "text": group["relevant_template"].format(i=index),
                })
            cases.append({
                "id": f"{kind}-{index}",
                "category": kind,
                "database_id": "primary",
                "query": group["query_template"].format(i=index),
                "relevant_doc_ids": relevant_ids,
                "documents": documents,
            })
        offset += group["count"]
    return cases


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def vector_results(case: dict[str, Any]) -> list[dict[str, Any]]:
    query_tokens = _tokens(case["query"])
    results = []
    for document in case["documents"]:
        if document.get("database_id") != case.get("database_id"):
            continue
        document_tokens = _tokens(document["text"])
        score = len(query_tokens & document_tokens) / max(1, len(query_tokens))
        results.append({"doc_id": document["doc_id"], "score": score})
    return sorted(results, key=lambda item: (-item["score"], item["doc_id"]))


def rerank_results(case: dict[str, Any]) -> list[dict[str, Any]]:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("rerank mode requires DASHSCOPE_API_KEY")
    from app.core.config import Settings
    from app.rag.reranker import DashScopeReranker

    documents = [
        document for document in case["documents"]
        if document.get("database_id") == case.get("database_id")
    ]
    outcome = DashScopeReranker(Settings()).rerank(
        case["query"], [document["text"] for document in documents], len(documents)
    )
    if outcome.degraded:
        raise RuntimeError(f"reranker failed: {outcome.error_code}")
    return [
        {"doc_id": documents[item.index]["doc_id"], "score": item.score}
        for item in outcome.items
    ]


def evaluate_cases(
    cases: Iterable[dict[str, Any]], threshold: float, k: int = 5
) -> dict[str, Any]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    cases = list(cases)
    for case in cases:
        relevant = set(case["relevant_doc_ids"])
        ranked = case.get("results", [])
        accepted = [item for item in ranked if item["score"] >= threshold]
        found = any(item["doc_id"] in relevant for item in accepted)
        predicted = bool(accepted)
        if relevant:
            counts["tp" if found else "fn"] += 1
        else:
            counts["fp" if predicted else "tn"] += 1

        top = ranked[:k]
        hits = sum(item["doc_id"] in relevant for item in top)
        precisions.append(hits / k)
        if relevant:
            recalls.append(hits / len(relevant))
            reciprocal_ranks.append(next(
                (1 / rank for rank, item in enumerate(ranked, 1) if item["doc_id"] in relevant),
                0.0,
            ))
    positives = counts["tp"] + counts["fn"]
    negatives = counts["fp"] + counts["tn"]
    total = len(cases)
    return {
        "threshold": threshold,
        "counts": counts,
        "hit_rate": counts["tp"] / positives if positives else 0.0,
        "false_hit_rate": counts["fp"] / negatives if negatives else 0.0,
        "false_rejection_rate": counts["fn"] / positives if positives else 0.0,
        "precision_at_k": sum(precisions) / total if total else 0.0,
        "recall_at_k": sum(recalls) / positives if positives else 0.0,
        "mrr": sum(reciprocal_ranks) / positives if positives else 0.0,
        "k": k,
    }


def calibrate(
    cases: Iterable[dict[str, Any]], thresholds: Iterable[float],
    max_fhr: float = .05, max_frr: float = .20, k: int = 5,
) -> dict[str, Any]:
    grid = [evaluate_cases(cases, threshold, k) for threshold in thresholds]
    feasible = [
        item for item in grid
        if item["false_hit_rate"] <= max_fhr
        and item["false_rejection_rate"] <= max_frr
    ]
    if not feasible:
        raise ValueError("no threshold satisfies FHR/FRR constraints")
    selected = max(feasible, key=lambda item: (item["hit_rate"], -item["threshold"]))
    return {"selected_threshold": selected["threshold"], "selected_metrics": selected, "grid": grid}


def render_markdown(report: dict[str, Any]) -> str:
    calibration = report["calibration"]
    metrics = calibration["selected_metrics"]
    counts = metrics["counts"]
    return f"""# RAG Evaluation Report

- Mode: {report["mode"]}
- Cases: {report["case_count"]}
- Selected threshold: {calibration["selected_threshold"]:.3f}
- TP / FP / FN / TN: {counts["tp"]} / {counts["fp"]} / {counts["fn"]} / {counts["tn"]}
- Hit rate: {metrics["hit_rate"]:.4f}
- False hit rate: {metrics["false_hit_rate"]:.4f}
- False rejection rate: {metrics["false_rejection_rate"]:.4f}
- Precision@{metrics["k"]}: {metrics["precision_at_k"]:.4f}
- Recall@{metrics["k"]}: {metrics["recall_at_k"]:.4f}
- MRR: {metrics["mrr"]:.4f}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--mode", choices=("vector", "rerank"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fhr", type=float, default=.05)
    parser.add_argument("--max-frr", type=float, default=.20)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    cases = load_fixture(args.fixtures)
    scorer = vector_results if args.mode == "vector" else rerank_results
    for case in cases:
        case["results"] = scorer(case)
    calibration = calibrate(
        cases, [value / 20 for value in range(21)],
        max_fhr=args.max_fhr, max_frr=args.max_frr, k=args.k,
    )
    report = {"mode": args.mode, "case_count": len(cases), "calibration": calibration}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
