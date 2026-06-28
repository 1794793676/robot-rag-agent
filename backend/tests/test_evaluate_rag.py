import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_rag import (
    RERANK_GRID,
    VECTOR_GRID,
    calibrate_rerank_pairs,
    calibrate,
    evaluate_cases,
    load_fixture,
    render_markdown,
    vector_results,
    rerank_results,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rag_eval"
FIXTURE = FIXTURE_DIR


def test_fixture_has_required_coverage():
    cases = load_fixture(FIXTURE)
    assert len(cases) >= 100
    assert sum(not case["relevant_doc_ids"] for case in cases) >= 40
    assert sum(case.get("category") == "cross_database" for case in cases) >= 20
    assert len({case["case_id"] for case in cases}) == len(cases)
    required = {"case_id", "rag_database_id", "query", "answerable",
                "relevant_doc_ids", "relevant_chunk_ids", "tags", "expected_facts"}
    assert all(required <= case.keys() for case in cases)
    assert {"exact", "paraphrase", "table", "multichunk", "ambiguous",
            "hard_negative", "unrelated", "cross_database"} <= {
                tag for case in cases for tag in case["tags"]
            }


def test_metric_formulas_and_counts():
    cases = [
        {"id": "tp", "relevant_doc_ids": ["a"], "results": [{"doc_id": "a", "score": .9}]},
        {"id": "fn", "relevant_doc_ids": ["b"], "results": [{"doc_id": "x", "score": .8}]},
        {"id": "fp", "relevant_doc_ids": [], "results": [{"doc_id": "x", "score": .7}]},
        {"id": "tn", "relevant_doc_ids": [], "results": [{"doc_id": "x", "score": .1}]},
    ]
    metrics = evaluate_cases(cases, threshold=.5, k=1)
    assert metrics["counts"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert metrics["hit_rate"] == .5
    assert metrics["false_hit_rate"] == .5
    assert metrics["false_rejection_rate"] == .5
    assert metrics["precision_at_k"] == pytest.approx(.25)
    assert metrics["recall_at_k"] == pytest.approx(.5)
    assert metrics["mrr"] == pytest.approx(.5)


def test_calibration_obeys_constraints_and_is_deterministic():
    cases = [
        {"id": "p1", "relevant_doc_ids": ["a"], "results": [{"doc_id": "a", "score": .9}]},
        {"id": "p2", "relevant_doc_ids": ["b"], "results": [{"doc_id": "b", "score": .6}]},
        {"id": "n", "relevant_doc_ids": [], "results": [{"doc_id": "x", "score": .7}]},
    ]
    result = calibrate(cases, thresholds=[.5, .7, .8], max_fhr=0, max_frr=.5, k=1)
    assert result["selected_threshold"] == .8
    assert result["selected_metrics"]["false_hit_rate"] == 0
    assert result["selected_metrics"]["false_rejection_rate"] == .5
    assert len(result["grid"]) == 3


def test_decision_only_uses_final_top_k():
    case = {"relevant_doc_ids": ["a"], "relevant_chunk_ids": [], "tags": ["exact"],
            "results": [{"doc_id": "x", "score": .9}, {"doc_id": "a", "score": .8}]}
    metrics = evaluate_cases([case], threshold=.5, k=1)
    assert metrics["counts"]["fn"] == 1
    assert metrics["per_case"][0]["decision"] is False


def test_failed_calibration_uses_documented_default():
    cases = [{"relevant_doc_ids": ["a"], "relevant_chunk_ids": [], "tags": ["exact"],
              "results": []}]
    result = calibrate(cases, [.25], max_frr=0, default_threshold=.35)
    assert result["status"] == "failed"
    assert result["selected_threshold"] == .35


def test_mode_specific_grids_and_fake_reranker(monkeypatch):
    assert VECTOR_GRID == pytest.approx([.25, .30, .35, .40, .45, .50, .55, .60])
    assert RERANK_GRID == pytest.approx([.30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80])
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    case = load_fixture(FIXTURE)[0]
    assert rerank_results(case, fake=True) == rerank_results(case, fake=True)


def test_fake_reranker_changes_order_and_pair_calibration():
    case = {
        "case_id": "promotion", "tags": ["hard"], "rag_database_id": "primary",
        "query": "reset device", "relevant_doc_ids": ["relevant"],
        "relevant_chunk_ids": ["c-rel"],
        "documents": [
            {"doc_id": "decoy", "chunk_id": "c-decoy", "database_id": "primary",
             "text": "reset device", "fake_rerank_score": .1},
            {"doc_id": "relevant", "chunk_id": "c-rel", "database_id": "primary",
             "text": "factory restoration", "fake_rerank_score": .95},
        ],
    }
    reranked = rerank_results(case, fake=True, similarity_threshold=0)
    assert reranked[0]["doc_id"] == "relevant"
    result = calibrate_rerank_pairs([case], [.0], [.5], fake=True, k=1)
    assert result["status"] == "passed"
    assert result["selected_similarity_threshold"] == 0
    assert result["selected_rerank_threshold"] == .5


def test_vector_scoring_is_scoped_to_case_database():
    case = {
        "query": "secret payroll policy",
        "database_id": "primary",
        "documents": [
            {"doc_id": "foreign", "database_id": "other", "text": "secret payroll policy"},
            {"doc_id": "local", "database_id": "primary", "text": "device manual"},
        ],
    }
    assert [item["doc_id"] for item in vector_results(case)] == ["local"]


def test_reports_include_metrics():
    metrics = evaluate_cases([], .5)
    report = render_markdown({
        "mode": "vector", "model_id": "fake", "config_id": "test",
        "case_count": 4, "aggregate": metrics,
        "calibration": {"status": "passed", "selected_threshold": .5},
    })
    assert "RAG Evaluation Report" in report
    assert "Precision@5" in report
    assert "False hit rate" in report


def test_vector_cli_writes_json_and_markdown(tmp_path):
    output = tmp_path / "report"
    command = [
        sys.executable, "-m", "scripts.evaluate_rag",
        "--fixtures", str(FIXTURE), "--mode", "vector", "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "report.json").read_text())
    assert payload["case_count"] >= 100
    assert payload["per_case"]
    assert payload["per_category"]
    assert (output / "report.md").read_text().startswith("# RAG Evaluation Report")


def test_vector_cli_accepts_fixture_directory(tmp_path):
    output = tmp_path / "report"
    command = [
        sys.executable, "-m", "scripts.evaluate_rag",
        "--fixtures", "tests/fixtures/rag_eval",
        "--mode", "vector", "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=Path(__file__).parents[1], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()


def test_direct_script_cli_bootstraps_backend_imports(tmp_path):
    output = tmp_path / "direct-report"
    command = [
        ".venv/bin/python", "scripts/evaluate_rag.py",
        "--fixtures", "tests/fixtures/rag_eval",
        "--mode", "vector", "--output", str(output),
    ]
    completed = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "report.json").is_file()
