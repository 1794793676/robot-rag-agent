"""Reproducible offline RAG retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Direct execution sets sys.path[0] to backend/scripts rather than backend.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

VECTOR_GRID = [round(.25 + i * .05, 2) for i in range(8)]
RERANK_GRID = [round(.30 + i * .05, 2) for i in range(11)]
DEFAULT_THRESHOLDS = {"vector": .35, "rerank": .50}


def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    root = path if path.is_dir() else path.parent
    payload = json.loads((path / "cases.json" if path.is_dir() else path).read_text())
    if isinstance(payload, list):
        cases = payload
    else:
        sources = {
            name: (root / relative).read_text()
            for name, relative in payload.get("documents", {}).items()
        }
        cases = []
        offset = 0
        positive_tags = ["exact", "paraphrase", "table", "multichunk", "ambiguous"]
        for group in payload["groups"]:
            for i in range(group["count"]):
                number = offset + i
                kind = group["kind"]
                tag = positive_tags[i % len(positive_tags)] if kind == "positive" else (
                    "cross_database" if kind == "cross_database"
                    else ("hard_negative" if i % 2 == 0 else "unrelated")
                )
                doc_id, chunk_id = f"doc-{number}", f"chunk-{number}"
                database_id = "other" if kind == "cross_database" else "primary"
                text = sources.get(tag, group["distractor_template"].format(i=i))
                relevant_docs = [doc_id] if kind == "positive" else []
                relevant_chunks = [chunk_id] if kind == "positive" else []
                if kind == "positive":
                    text = f'{sources.get(tag, group["relevant_template"])} Device code {i}.'
                cases.append({
                    "case_id": f"{kind}-{i}",
                    "category": kind,
                    "rag_database_id": "primary",
                    "query": group["query_template"].format(i=i),
                    "answerable": kind == "positive",
                    "relevant_doc_ids": relevant_docs,
                    "relevant_chunk_ids": relevant_chunks,
                    "tags": [tag],
                    "expected_facts": [f"device code {i}"] if kind == "positive" else [],
                    "documents": [{"doc_id": doc_id, "chunk_id": chunk_id,
                                   "database_id": database_id, "text": text}],
                })
            offset += group["count"]
    for case in cases:
        for document in case.get("documents", []):
            if "path" in document and "text" not in document:
                document["text"] = (root / document["path"]).read_text()
    return cases


def vector_results(case: dict[str, Any]) -> list[dict[str, Any]]:
    from app.core.config import Settings
    from app.rag.embedder import Embedder

    documents = [doc for doc in case["documents"]
                 if doc.get("database_id") == case.get("rag_database_id", case.get("database_id"))]
    embedder = Embedder(Settings())
    query = np.asarray(embedder._fake_embedding(case["query"]))
    results = []
    for doc in documents:
        score = float(np.dot(query, np.asarray(embedder._fake_embedding(doc["text"]))))
        results.append({"doc_id": doc["doc_id"], "chunk_id": doc.get("chunk_id"),
                        "score": max(0.0, min(1.0, score))})
    return sorted(results, key=lambda item: (-item["score"], item["doc_id"]))


def rerank_results(case: dict[str, Any], fake: bool = False, candidate_k: int = 20) -> list[dict[str, Any]]:
    candidates = vector_results(case)[:candidate_k]
    documents_by_id = {doc["doc_id"]: doc for doc in case["documents"]}
    if fake:
        return sorted(candidates, key=lambda item: (-item["score"], item["doc_id"]))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("live rerank mode requires DASHSCOPE_API_KEY (use --fake-reranker offline)")
    from app.core.config import Settings
    from app.rag.reranker import DashScopeReranker

    docs = [documents_by_id[item["doc_id"]] for item in candidates]
    outcome = DashScopeReranker(Settings()).rerank(
        case["query"], [doc["text"] for doc in docs], len(docs))
    if outcome.degraded:
        raise RuntimeError(f"reranker failed: {outcome.error_code}")
    return [{"doc_id": docs[item.index]["doc_id"],
             "chunk_id": docs[item.index].get("chunk_id"), "score": item.score}
            for item in outcome.items]


def evaluate_cases(cases: Iterable[dict[str, Any]], threshold: float, k: int = 5,
                   include_categories: bool = True) -> dict[str, Any]:
    cases = list(cases)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    precision, recall, rr, details = [], [], [], []
    for case in cases:
        relevant_docs = set(case.get("relevant_doc_ids", []))
        relevant_chunks = set(case.get("relevant_chunk_ids", []))
        ranked = case.get("results", [])
        top = ranked[:k]
        is_relevant = lambda item: (
            item.get("doc_id") in relevant_docs
            or (item.get("chunk_id") is not None and item.get("chunk_id") in relevant_chunks)
        )
        accepted = [item for item in top if item["score"] >= threshold]
        expected = bool(relevant_docs or relevant_chunks)
        decision = any(is_relevant(item) for item in accepted) if expected else bool(accepted)
        counts[("tp" if decision else "fn") if expected else ("fp" if decision else "tn")] += 1
        hits = sum(is_relevant(item) for item in top)
        precision.append(hits / k)
        if expected:
            denominator = max(len(relevant_docs), len(relevant_chunks), 1)
            recall.append(hits / denominator)
            rr.append(next((1 / rank for rank, item in enumerate(top, 1) if is_relevant(item)), 0.0))
        details.append({
            "case_id": case.get("case_id", case.get("id")),
            "tags": case.get("tags", [case.get("category", "uncategorized")]),
            "decision": decision, "expected_answerable": expected,
            "ranked_candidates": [dict(item, rank=rank, accepted=item["score"] >= threshold)
                                  for rank, item in enumerate(top, 1)],
        })
    positives, negatives, total = counts["tp"] + counts["fn"], counts["fp"] + counts["tn"], len(cases)
    metrics = {
        "threshold": threshold, "counts": counts,
        "hit_rate": counts["tp"] / positives if positives else 0.0,
        "false_hit_rate": counts["fp"] / negatives if negatives else 0.0,
        "false_rejection_rate": counts["fn"] / positives if positives else 0.0,
        "precision_at_k": sum(precision) / total if total else 0.0,
        "recall_at_k": sum(recall) / positives if positives else 0.0,
        "mrr": sum(rr) / positives if positives else 0.0, "k": k,
        "per_case": details,
    }
    if include_categories:
        tags = sorted({tag for case in cases for tag in case.get("tags", [])})
        metrics["per_category"] = {
            tag: evaluate_cases([case for case in cases if tag in case.get("tags", [])],
                                threshold, k, False)
            for tag in tags
        }
    return metrics


def calibrate(cases: Iterable[dict[str, Any]], thresholds: Iterable[float],
              max_fhr: float = .05, max_frr: float = .15, k: int = 5,
              default_threshold: float = .35) -> dict[str, Any]:
    cases = list(cases)
    grid = [evaluate_cases(cases, threshold, k, False) for threshold in thresholds]
    feasible = [m for m in grid if m["false_hit_rate"] <= max_fhr
                and m["false_rejection_rate"] <= max_frr]
    if feasible:
        selected = max(feasible, key=lambda m: (m["hit_rate"], m["mrr"],
                                                -m["false_hit_rate"], -m["threshold"]))
        status = "passed"
    else:
        selected = evaluate_cases(cases, default_threshold, k, False)
        status = "failed"
    return {"status": status, "selected_threshold": selected["threshold"],
            "selected_metrics": selected, "grid": grid,
            "constraints": {"max_fhr": max_fhr, "max_frr": max_frr},
            "default_threshold": default_threshold}


def render_markdown(report: dict[str, Any]) -> str:
    c, m = report["calibration"], report["aggregate"]
    counts = m["counts"]
    return f"""# RAG Evaluation Report

- Mode: {report["mode"]}
- Model/config: {report["model_id"]} / {report["config_id"]}
- Cases: {report["case_count"]}
- Calibration: {c["status"]}; threshold {c["selected_threshold"]:.2f}
- TP / FP / FN / TN: {counts["tp"]} / {counts["fp"]} / {counts["fn"]} / {counts["tn"]}
- Hit rate: {m["hit_rate"]:.4f}
- False hit rate: {m["false_hit_rate"]:.4f}
- False rejection rate: {m["false_rejection_rate"]:.4f}
- Precision@{m["k"]}: {m["precision_at_k"]:.4f}
- Recall@{m["k"]}: {m["recall_at_k"]:.4f}
- MRR: {m["mrr"]:.4f}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--mode", choices=("vector", "rerank"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fhr", type=float, default=.05)
    parser.add_argument("--max-frr", type=float, default=.15)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fake-reranker", action="store_true")
    args = parser.parse_args(argv)
    cases = load_fixture(args.fixtures)
    vector_baseline = None
    if args.mode == "rerank":
        for case in cases:
            case["results"] = vector_results(case)
        vector_calibration = calibrate(
            cases, VECTOR_GRID, args.max_fhr, args.max_frr, args.k,
            DEFAULT_THRESHOLDS["vector"])
        vector_baseline = evaluate_cases(
            cases, vector_calibration["selected_threshold"], args.k)
    for case in cases:
        case["results"] = (vector_results(case) if args.mode == "vector"
                           else rerank_results(case, args.fake_reranker))
    default = DEFAULT_THRESHOLDS[args.mode]
    calibration = calibrate(cases, VECTOR_GRID if args.mode == "vector" else RERANK_GRID,
                            args.max_fhr, args.max_frr, args.k, default)
    aggregate = evaluate_cases(cases, calibration["selected_threshold"], args.k)
    comparison = None
    if vector_baseline is not None:
        comparison = {
            metric: aggregate[metric] - vector_baseline[metric]
            for metric in ("hit_rate", "mrr", "false_hit_rate",
                           "false_rejection_rate", "precision_at_k", "recall_at_k")
        }
    report = {
        "dataset_version": "1.0", "mode": args.mode, "case_count": len(cases),
        "model_id": "deterministic-embedder" if args.mode == "vector"
        else ("fake-reranker" if args.fake_reranker else "dashscope-reranker"),
        "config_id": f"top_k={args.k}", "calibration": calibration,
        "aggregate": aggregate, "per_case": aggregate["per_case"],
        "per_category": aggregate["per_category"], "comparison_delta": comparison,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "report.md").write_text(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
