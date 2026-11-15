"""Shared helpers: project paths and a logger that writes to the terminal and logs/<name>_<time>.log.

Console output goes through tqdm.write so it never breaks a live progress bar.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
CROPS_DIR = DATA_DIR / "crops"
SPLITS_DIR = DATA_DIR / "splits"
ZIP_DIR = PROJECT_ROOT / "zip data"
RUNS_DIR = PROJECT_ROOT / "runs"
PIPELINE_OK = DATA_DIR / "pipeline_ok.json"


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logger(name: str) -> tuple[logging.Logger, Path]:
    """Return (logger, log_path). Every message goes to the terminal and the log file."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = _TqdmHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, log_path
