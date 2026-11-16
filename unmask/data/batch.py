"""Batch bookkeeping: the download plan, part roles, DONE.json, zip deletion after the smoke test (PLAN 6, 7.1)."""
import json
import shutil
from pathlib import Path

from ..common import CROPS_DIR, PIPELINE_OK, ZIP_DIR
from .dfdc_zip import find_part_zips

BATCHES = [(0, 1), (2, 8), (3, 9), (4, 5), (6, 7), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)]

# Part roles (PLAN 7.1, 4.3). Part 09 is the DFDC test set: never loaded for training or validation.
TRAIN_PARTS = tuple(range(0, 8)) + tuple(range(10, 20))
DEV_PARTS = (0, 1, 2, 3, 4, 5)  # Stage A development set (PLAN 10.1)
VAL_PART = 8
TEST_PART = 9
EVAL_PARTS = (VAL_PART, TEST_PART)


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
    """Names the next missing part, in batch-table order. Zips are downloaded one at a time (owner, 16 Nov):
    start the next download only with at least 16 GB free."""
    for batch in BATCHES:
        missing = [p for p in batch if p not in done]
        if missing:
            return (f'download dfdc_train_part_{missing[0]:02d}.zip into "zip data\\" (one zip at a time, start '
                    f"only with >= 16 GB free), then run: python scripts\\process_batch.py")
    return "DFDC done. Run the finals queue (section 10.2)."


def check_split_parts(train: list[int] | tuple[int, ...], val: list[int] | tuple[int, ...]) -> None:
    """Raise if training touches the validation or test part, or validation touches the test part."""
    bad_train = sorted(set(train) - set(TRAIN_PARTS))
    if bad_train:
        raise ValueError(f"training split includes non-training parts {bad_train}; train parts are 00-07 and 10-19 only")
    if TEST_PART in set(val):
        raise ValueError(f"validation split includes the test part {TEST_PART:02d}")


def train_parts(done: set[int]) -> list[int]:
    """Extracted parts that may be trained on (00-07 and 10-19)."""
    return sorted(p for p in done if p in TRAIN_PARTS)


def val_parts(done: set[int]) -> list[int]:
    """[8] once part 08 is extracted, else [] (keep using data\\splits\\temp_val.json until then)."""
    return [VAL_PART] if VAL_PART in done else []


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
