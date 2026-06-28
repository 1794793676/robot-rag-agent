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
                documents = [{"doc_id": doc_id, "chunk_id": chunk_id,
                              "database_id": database_id, "text": text,
                              "fake_rerank_score": .95 if kind == "positive" else .1}]
                if kind == "positive" and i == 0:
                    documents[0]["text"] = "Factory restoration procedure."
                    for decoy in range(6):
                        documents.append({
                            "doc_id": f"decoy-{decoy}", "chunk_id": f"decoy-chunk-{decoy}",
                            "database_id": "primary",
                            "text": f"How to reset device code {i}, overview {decoy}.",
                            "fake_rerank_score": .05 + decoy / 100,
                        })
                if kind == "positive" and tag == "multichunk":
                    second_chunk = f"{chunk_id}-part2"
                    relevant_chunks.append(second_chunk)
                    documents.append({
                        "doc_id": doc_id, "chunk_id": second_chunk,
                        "database_id": "primary",
                        "text": "Reconnect power while continuing to hold reset.",
                        "fake_rerank_score": .9,
                    })
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
                    "documents": documents,
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
        result = {"doc_id": doc["doc_id"], "chunk_id": doc.get("chunk_id"),
                  "score": max(0.0, min(1.0, score))}
        if "fake_rerank_score" in doc:
            result["fake_rerank_score"] = doc["fake_rerank_score"]
        results.append(result)
    return sorted(results, key=lambda item: (-item["score"], item["doc_id"]))


def rerank_results(case: dict[str, Any], fake: bool = False, candidate_k: int = 20,
                   similarity_threshold: float = 0.0) -> list[dict[str, Any]]:
    candidates = [item for item in vector_results(case)
                  if item["score"] >= similarity_threshold][:candidate_k]
    if not candidates:
        return []
    documents_by_id = {
        (doc["doc_id"], doc.get("chunk_id")): doc for doc in case["documents"]
    }
    if fake:
        reranked = [dict(item, vector_score=item["score"],
                         score=item.get("fake_rerank_score", item["score"]))
                    for item in candidates]
        return sorted(reranked, key=lambda item: (-item["score"], item["doc_id"]))
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("live rerank mode requires DASHSCOPE_API_KEY (use --fake-reranker offline)")
    from app.core.config import Settings
    from app.rag.reranker import DashScopeReranker

    docs = [
        documents_by_id[(item["doc_id"], item.get("chunk_id"))]
        for item in candidates
    ]
    outcome = DashScopeReranker(Settings()).rerank(
        case["query"], [doc["text"] for doc in docs], len(docs))
    if outcome.degraded:
        raise RuntimeError(f"reranker failed: {outcome.error_code}")
    return [{
        "doc_id": docs[item.index]["doc_id"],
        "chunk_id": docs[item.index].get("chunk_id"),
        "score": item.score,
        "vector_score": candidates[item.index]["score"],
    } for item in outcome.items]


def calibrate_rerank_pairs(
    cases: Iterable[dict[str, Any]], similarity_thresholds: Iterable[float],
    rerank_thresholds: Iterable[float], fake: bool = False, candidate_k: int = 20,
    max_fhr: float = .05, max_frr: float = .15, k: int = 5,
) -> dict[str, Any]:
    cases = list(cases)
    grid = []
    rerank_cache = {
        case["case_id"]: rerank_results(case, fake, candidate_k, 0.0)
        for case in cases
    }
    for similarity_threshold in similarity_thresholds:
        for case in cases:
            case["results"] = [
                item for item in rerank_cache[case["case_id"]]
                if item.get("vector_score", item["score"]) >= similarity_threshold
            ]
        for rerank_threshold in rerank_thresholds:
            metrics = evaluate_cases(cases, rerank_threshold, k, False)
            grid.append({
                "similarity_threshold": similarity_threshold,
                "rerank_threshold": rerank_threshold, "metrics": metrics,
            })
    feasible = [item for item in grid
                if item["metrics"]["false_hit_rate"] <= max_fhr
                and item["metrics"]["false_rejection_rate"] <= max_frr]
    if feasible:
        selected = max(feasible, key=lambda item: (
            item["metrics"]["hit_rate"], item["metrics"]["mrr"],
            -item["metrics"]["false_hit_rate"], -item["similarity_threshold"],
            -item["rerank_threshold"]))
        status = "passed"
    else:
        selected = next((item for item in grid
                         if item["similarity_threshold"] == DEFAULT_THRESHOLDS["vector"]
                         and item["rerank_threshold"] == DEFAULT_THRESHOLDS["rerank"]),
                        grid[0])
        status = "failed"
    selected_results = {
        case["case_id"]: [
            item for item in rerank_cache[case["case_id"]]
            if item.get("vector_score", item["score"])
            >= selected["similarity_threshold"]
        ]
        for case in cases
    }
    return {
        "status": status,
        "selected_similarity_threshold": selected["similarity_threshold"],
        "selected_rerank_threshold": selected["rerank_threshold"],
        "selected_threshold": selected["rerank_threshold"],
        "selected_metrics": selected["metrics"], "grid": grid,
        "constraints": {"max_fhr": max_fhr, "max_frr": max_frr},
        "_selected_results": selected_results,
    }


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
- Model/config: {report["model_id"]} / {report.get("config", report.get("config_id"))}
- Cases: {report["case_count"]}
- Calibration: {c["status"]}; threshold {c["selected_threshold"]:.2f}
- TP / FP / FN / TN: {counts["tp"]} / {counts["fp"]} / {counts["fn"]} / {counts["tn"]}
- Hit rate: {m["hit_rate"]:.4f}
- False hit rate: {m["false_hit_rate"]:.4f}
- False rejection rate: {m["false_rejection_rate"]:.4f}
- Precision@{m["k"]}: {m["precision_at_k"]:.4f}
- Recall@{m["k"]}: {m["recall_at_k"]:.4f}
- MRR: {m["mrr"]:.4f}
""" + "\n## Per-category\n\n| Category | Hit | FHR | FRR | MRR |\n|---|---:|---:|---:|---:|\n" + "".join(
        f'| {tag} | {value["hit_rate"]:.3f} | {value["false_hit_rate"]:.3f} | '
        f'{value["false_rejection_rate"]:.3f} | {value["mrr"]:.3f} |\n'
        for tag, value in report.get("per_category", m.get("per_category", {})).items()
    ) + "\n## Per-case\n\n| Case | Decision | Ranked candidates |\n|---|---|---|\n" + "".join(
        f'| {case["case_id"]} | {case["decision"]} | ' +
        ", ".join(f'{item["rank"]}:{item["doc_id"]}={item["score"]:.3f}'
                  for item in case["ranked_candidates"]) + " |\n"
        for case in report.get("per_case", m.get("per_case", []))
    ) + ("\n## Comparison deltas\n\n" + json.dumps(report.get("comparison_delta"))
         if report.get("comparison_delta") is not None else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--mode", choices=("vector", "rerank"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fhr", type=float, default=.05)
    parser.add_argument("--max-frr", type=float, default=.15)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fake-reranker", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=20)
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
    if args.mode == "vector":
        for case in cases:
            case["results"] = vector_results(case)
        calibration = calibrate(cases, VECTOR_GRID, args.max_fhr, args.max_frr,
                                args.k, DEFAULT_THRESHOLDS["vector"])
    else:
        calibration = calibrate_rerank_pairs(
            cases, VECTOR_GRID, RERANK_GRID, args.fake_reranker,
            args.candidate_k, args.max_fhr, args.max_frr, args.k)
        selected_results = calibration.pop("_selected_results")
        for case in cases:
            case["results"] = selected_results[case["case_id"]]
    aggregate = evaluate_cases(cases, calibration["selected_threshold"], args.k)
    comparison = None
    if vector_baseline is not None:
        comparison = {
            metric: aggregate[metric] - vector_baseline[metric]
            for metric in ("hit_rate", "mrr", "false_hit_rate",
                           "false_rejection_rate", "precision_at_k", "recall_at_k")
        }
    if args.mode == "rerank" and not args.fake_reranker:
        from app.core.config import Settings
        model_id = Settings().rerank_model
    else:
        model_id = ("deterministic-embedder" if args.mode == "vector"
                    else "fake-reranker")
    report = {
        "dataset_version": "1.0", "mode": args.mode, "case_count": len(cases),
        "model_id": model_id,
        "config": {"candidate_k": args.candidate_k, "top_k": args.k,
                   "defaults": DEFAULT_THRESHOLDS, "similarity_grid": VECTOR_GRID,
                   "rerank_grid": RERANK_GRID},
        "config_id": f"candidate_k={args.candidate_k};top_k={args.k}",
        "calibration": calibration,
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
