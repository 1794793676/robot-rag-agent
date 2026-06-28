from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures" / "rag_eval"


def test_rerank_cli_requires_dashscope_key(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("DASHSCOPE_API_KEY", None)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_rag.py",
            "--fixtures",
            str(FIXTURES),
            "--mode",
            "rerank",
            "--output",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert completed.returncode != 0
    assert "DASHSCOPE_API_KEY" in completed.stderr
