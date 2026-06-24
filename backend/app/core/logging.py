"""Small, production-friendly logging setup."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> None:
    """Configure root logging once without adding heavy dependencies."""

    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    root.setLevel(logging.INFO)

    for logger_name, filename in {
        "agent": "agent.log",
        "webrtc": "webrtc.log",
        "tool_calls": "tool_calls.log",
        "errors": "errors.log",
    }.items():
        logger = logging.getLogger(logger_name)
        if any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
            continue
        handler = RotatingFileHandler(
            log_dir / filename,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = True
