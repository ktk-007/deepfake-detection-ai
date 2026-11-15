"""Grouped splits: a real video and all its fakes always land on the same side (PLAN 3.4, 6.1)."""
import json
import random
from pathlib import Path


def group_key(part: int | str, real: str) -> str:
    return f"{int(part):02d}/{real}"


def groups_from_rows(rows: list[dict]) -> list[str]:
    """Group keys (part/real) of every real video with status ok."""
    return sorted(group_key(r["part"], r["video"]) for r in rows if r["label"] == "REAL" and r["status"] == "ok")


def grouped_split(groups: list[str], val_frac: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """(train, val) group keys. Deterministic for a given set of groups and seed."""
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    n_val = max(1, round(len(g) * val_frac)) if len(g) > 1 else 0
    return sorted(g[n_val:]), sorted(g[:n_val])


def save_split(path: Path, train: list[str], val: list[str], **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(train=train, val=val, **extra), indent=1))


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())
