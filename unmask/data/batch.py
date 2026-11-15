"""Batch bookkeeping: the download plan, DONE.json, zip deletion after the smoke test (PLAN 6.1, 7.2)."""
import json
import shutil
from pathlib import Path

from ..common import CROPS_DIR, PIPELINE_OK, ZIP_DIR
from .dfdc_zip import find_part_zips

BATCHES = [(0, 1), (48, 49), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def part_dir(part: int, crops_dir: Path = CROPS_DIR) -> Path:
    return crops_dir / f"part_{part:02d}"


def done_info(part: int, crops_dir: Path = CROPS_DIR) -> dict | None:
    p = part_dir(part, crops_dir) / "DONE.json"
    return json.loads(p.read_text()) if p.exists() else None


def done_parts(crops_dir: Path = CROPS_DIR) -> set[int]:
    out = set()
    if crops_dir.is_dir():
        for d in crops_dir.glob("part_*"):
            if (d / "DONE.json").exists():
                try:
                    out.add(int(d.name.split("_")[1]))
                except ValueError:
                    pass
    return out


def next_message(done: set[int]) -> str:
    for a, b in BATCHES:
        if a not in done or b not in done:
            return (f'download dfdc_train_part_{a:02d}.zip and dfdc_train_part_{b:02d}.zip into "zip data\\", '
                    f"then run: python scripts\\process_batch.py")
    return "all planned batches are done. Run the learning-curve job (E5) to decide on more parts."


def train_parts(done: set[int]) -> list[int]:
    return sorted(p for p in done if p not in BATCHES[1])


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def delete_done_zips(log, zip_dir: Path = ZIP_DIR, crops_dir: Path = CROPS_DIR,
                     pipeline_ok: Path = PIPELINE_OK) -> tuple[list[str], str]:
    """Delete dfdc_train_part_NN.zip for every part with DONE.json, only if pipeline_ok.json exists.

    Returns (deleted descriptions, status text). Touches only files matching the zip name pattern in zip_dir.
    """
    if not pipeline_ok.exists():
        return [], "kept, smoke test not passed yet"
    done = done_parts(crops_dir)
    deleted = []
    for part, path in find_part_zips(zip_dir).items():
        if part not in done:
            continue
        info = done_info(part, crops_dir) or {}
        if not info.get("gates_pass"):
            continue
        size = path.stat().st_size
        path.unlink()
        msg = f"{path.name} ({size / 2**30:.2f} GB)"
        log.info(f"deleted {msg}")
        deleted.append(msg)
    return deleted, (", ".join(deleted) if deleted else "none to delete")
