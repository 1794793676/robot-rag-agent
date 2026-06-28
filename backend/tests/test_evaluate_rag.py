import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_rag import (
    calibrate,
    evaluate_cases,
    load_fixture,
    render_markdown,
    vector_results,
)


FIXTURE = Path(__file__).parent / "fixtures" / "rag_evaluation.json"


def test_fixture_has_required_coverage():
    cases = load_fixture(FIXTURE)
    assert len(cases) >= 100
    assert sum(not case["relevant_doc_ids"] for case in cases) >= 40
    assert sum(case.get("category") == "cross_database" for case in cases) >= 20
    assert len({case["id"] for case in cases}) == len(cases)


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
    report = render_markdown({"mode": "vector", "case_count": 4, "calibration": {
        "selected_threshold": .5,
        "selected_metrics": evaluate_cases([], .5),
        "grid": [],
    }})
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
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["case_count"] >= 100
    assert output.with_suffix(".md").read_text().startswith("# RAG Evaluation Report")
