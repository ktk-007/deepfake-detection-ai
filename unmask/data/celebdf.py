"""Celeb-DF v2 official test list -> face crops (PLAN 7.5). Test only: never trained or tuned on.

Only the videos in List_of_testing_videos.txt are processed; the count is read from the file. Each video is
streamed from the zip to a temp file (deleted at once). No pairing: faces are detected per video with the same
face-choice rule as DFDC, 32 evenly spaced frames, the same 30% margin, 224 px, JPEG 95.
Label: from the folder (Celeb-synthesis = FAKE; Celeb-real, YouTube-real = REAL), checked against the list's
own 0/1 column (gate C5). index.csv uses the DFDC columns, with part = "celebdf".
"""
import random
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from tqdm import tqdm

from .dfdc_zip import read_frames
from .extract import save_crops
from .faces import crops_for
from .index import IndexWriter, read_index, video_dir

LIST_NAME = "List_of_testing_videos.txt"
PART = "celebdf"


@dataclass
class CelebConfig:
    frames: int = 32
    margin: float = 0.30
    size: int = 224
    quality: int = 95
    min_frames_frac: float = 0.5


def folder_label(path: str) -> str:
    top = PurePosixPath(path).parts[0].lower()
    if "synthesis" in top:
        return "FAKE"
    if "real" in top:
        return "REAL"
    raise ValueError(f"unknown Celeb-DF folder in {path!r}")


def find_list(names: list[str]) -> tuple[str, str]:
    """(member name of the test list, prefix of the dataset root inside the zip)."""
    hits = [n for n in names if PurePosixPath(n).name == LIST_NAME]
    if len(hits) != 1:
        raise SystemExit(f"expected one {LIST_NAME} in the zip, found {len(hits)}")
    parent = str(PurePosixPath(hits[0]).parent)
    return hits[0], "" if parent == "." else parent + "/"


def parse_list(text: str) -> list[tuple[int, str]]:
    """Lines '<0|1> <folder>/<file>.mp4' -> [(flag, path)]; blank lines skipped, duplicates kept once."""
    out, seen = [], set()
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 2 or parts[0] not in ("0", "1"):
            raise ValueError(f"unexpected line in {LIST_NAME}: {line!r}")
        if parts[1] not in seen:
            seen.add(parts[1])
            out.append((int(parts[0]), parts[1]))
    return out


def list_flag_meaning(entries: list[tuple[int, str]]) -> dict:
    """Which list flag goes with which folder label; consistent means each flag maps to one label only."""
    pairs = Counter((flag, folder_label(p)) for flag, p in entries)
    by_flag = {f: {lab for (ff, lab) in pairs if ff == f} for f in {f for f, _ in pairs}}
    return {"pairs": {f"{f}->{lab}": n for (f, lab), n in sorted(pairs.items())},
            "consistent": all(len(v) == 1 for v in by_flag.values()) and
            len({next(iter(v)) for v in by_flag.values()}) == len(by_flag)}


def extract_celebdf(zip_path: Path, out_dir: Path, cfg: CelebConfig, detector, log=None, limit: int = 0,
                    reader=read_frames, show_bar: bool = True) -> dict:
    say = log.info if log else print
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    done = {r["video"] for r in read_index(index_path)}
    stats, t0 = Counter(), time.time()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        list_member, prefix = find_list(names)
        raw = zf.read(list_member)
        (out_dir / LIST_NAME).write_bytes(raw)
        entries = parse_list(raw.decode("utf-8", errors="replace"))
        members = set(names)
        stems = Counter(PurePosixPath(p).stem for _, p in entries)
        clash = [s for s, n in stems.items() if n > 1]
        if clash:
            raise SystemExit(f"two listed videos share a file name, crops would collide: {clash[:3]}")
        todo = [(f, p) for f, p in entries if p not in done]
        if limit:
            todo = todo[:limit]
        say(f"celebdf: {len(entries)} videos in {LIST_NAME}; {len(done)} already done, {len(todo)} to go")
        stats["listed"] = len(entries)
        with IndexWriter(index_path) as writer:
            for flag, path in tqdm(todo, unit="video", dynamic_ncols=True, desc="celebdf", disable=not show_bar):
                label = folder_label(path)
                row = dict(video=path, label=label, original="", part=PART)
                if prefix + path not in members:
                    row.update(n_frames=0, status="missing")
                else:
                    frames, _ = reader(zf, prefix + path, cfg.frames)
                    if len(frames) < max(1, int(cfg.frames * cfg.min_frames_frac)):
                        row.update(n_frames=len(frames), status="read_error")
                    else:
                        boxes = detector(frames)
                        if boxes is None:
                            row.update(n_frames=len(frames), status="no_face")
                        else:
                            crops = crops_for(frames, boxes, frames[0].shape[:2], cfg.margin, cfg.size)
                            save_crops(crops, video_dir(out_dir, path), cfg.quality)
                            row.update(n_frames=len(crops), status="ok")
                stats[f"{label.lower()}_{row['status']}"] += 1
                writer.write_group([row])
    stats["seconds"] = round(time.time() - t0, 1)
    stats["config"] = asdict(cfg)
    stats["flags"] = list_flag_meaning(entries)
    return dict(stats)


def run_celeb_gates(out_dir: Path, rows: list[dict], listed: int, flags: dict, n_check: int = 200,
                    seed: int = 0, size: int = 224) -> list[tuple[str, bool, str]]:
    ok = [r for r in rows if r["status"] == "ok"]
    missing = sum(r["status"] == "missing" for r in rows)
    g = []
    g.append(("C1", len(rows) == listed and missing <= 0.01 * listed,
              f"{len(rows)}/{listed} listed videos have a row, {missing} missing from the zip (max 1%)"))
    g.append(("C2", len(ok) >= 0.95 * listed, f"ok {len(ok)}/{listed} = {100 * len(ok) / max(1, listed):.1f}% "
                                             f"(need >= 95%)"))
    short = [r["video"] for r in ok if len(list(video_dir(out_dir, r["video"]).glob("*.jpg"))) != int(r["n_frames"])]
    g.append(("C3", not short, f"{len(ok) - len(short)}/{len(ok)} ok videos have n_frames JPEGs"))
    jpgs = [p for r in ok for p in video_dir(out_dir, r["video"]).glob("*.jpg")]
    pick = random.Random(seed).sample(jpgs, min(n_check, len(jpgs)))
    bad = [p for p in pick if (im := cv2.imread(str(p))) is None or im.shape[:2] != (size, size)]
    g.append(("C4", bool(pick) and not bad, f"{len(pick) - len(bad)}/{len(pick)} random JPEGs decode at {size}x{size}"))
    labs = Counter(r["label"] for r in ok)
    g.append(("C5", labs["REAL"] > 0 and labs["FAKE"] > 0 and flags["consistent"],
              f"ok real {labs['REAL']}, fake {labs['FAKE']}; list flag vs folder {flags['pairs']}"
              f" ({'consistent' if flags['consistent'] else 'INCONSISTENT'})"))
    return g


def sample_sheet(out_dir: Path, rows: list[dict], n: int = 4, seed: int = 0, size: int = 160) -> Path | None:
    rng = random.Random(seed)
    tiles = []
    for lab in ("REAL", "FAKE"):
        ok = [r for r in rows if r["status"] == "ok" and r["label"] == lab]
        for r in rng.sample(ok, min(n, len(ok))):
            jp = sorted(video_dir(out_dir, r["video"]).glob("*.jpg"))
            im = cv2.resize(cv2.imread(str(jp[len(jp) // 2])), (size, size))
            cv2.putText(im, f"{lab} {PurePosixPath(r['video']).stem}"[:22], (3, 14), 0, 0.4, (0, 255, 255), 1)
            tiles.append(im)
    if not tiles:
        return None
    while len(tiles) % n:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + n]) for i in range(0, len(tiles), n)])
    path = out_dir / "_samples.jpg"
    cv2.imwrite(str(path), sheet)
    return path
