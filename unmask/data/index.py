"""index.csv for one extracted part. One row per video; rows of a group are written after the group finishes."""
import csv
from pathlib import Path

INDEX_FIELDS = ["video", "label", "original", "n_frames", "status", "score", "flagged", "noise", "part"]


def read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != INDEX_FIELDS:
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {INDEX_FIELDS}")
        return list(reader)


def done_groups(rows: list[dict]) -> set[str]:
    """Real videos whose group is complete (the real's row is written last-in-group with the others)."""
    return {r["video"] for r in rows if r["label"] == "REAL"}


class IndexWriter:
    def __init__(self, path: Path):
        new = not path.exists()
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=INDEX_FIELDS)
        if new:
            self.w.writeheader()
            self.f.flush()

    def write_group(self, rows: list[dict]):
        for r in rows:
            self.w.writerow({k: r.get(k, "") for k in INDEX_FIELDS})
        self.f.flush()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def video_dir(out_dir: Path, video: str) -> Path:
    return out_dir / Path(video).stem
