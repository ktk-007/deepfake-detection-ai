"""WildDeepfake test split: tar files of face-cropped PNGs -> 32-frame JPEG clips (PLAN 7.3, docs/data-notes.md).

Layout inside each tar (uncompressed despite the .tar.gz name): ./<tar id>/<fake|real>/<sequence id>/<frame>.png.
One clip = one sequence folder. Clip id (and folder name) is <class>_<tar id>_<sequence id>, e.g. fake_1_33.
Tars are seekable, so only the selected members are read, into memory. Nothing is extracted to disk.
"""
import csv
import random
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .dfdc_zip import evenly_spaced

WILD_FIELDS = ["video", "label", "n_frames", "source_tar", "n_source_frames", "status"]
CLASSES = (("fake_test", "FAKE"), ("real_test", "REAL"))


@dataclass
class WildConfig:
    frames: int = 32
    size: int = 224
    quality: int = 95
    short_below: int = 8


def list_tars(root: Path) -> list[tuple[Path, str]]:
    """[(tar path, label)] sorted by class then numeric tar id."""
    out = []
    for folder, label in CLASSES:
        tars = [p for p in (root / folder).glob("*.tar*") if p.name.split(".")[0].isdigit()]
        out += [(p, label) for p in sorted(tars, key=lambda p: int(p.name.split(".")[0]))]
    return out


def tar_key(path: Path) -> str:
    """source_tar value, e.g. fake_test/1.tar.gz."""
    return f"{path.parent.name}/{path.name}"


def sequences(tf: tarfile.TarFile, label: str) -> tuple[dict[str, list[tuple[int, tarfile.TarInfo]]], int, int]:
    """Group PNG members by sequence. Returns ({clip id: [(frame no, member)] sorted}, n_png, n_skipped)."""
    cls = label.lower()
    seqs, n_png, skipped = {}, 0, 0
    for m in tf.getmembers():
        if not m.isfile():
            continue
        parts = PurePosixPath(m.name).parts
        parts = parts[1:] if parts and parts[0] == "." else parts
        ok = (len(parts) == 4 and parts[1] == cls and parts[0].isdigit() and parts[2].isdigit()
              and parts[3].lower().endswith(".png") and parts[3][:-4].isdigit())
        if not ok:
            skipped += 1
            continue
        n_png += 1
        seqs.setdefault(f"{cls}_{parts[0]}_{parts[2]}", []).append((int(parts[3][:-4]), m))
    for v in seqs.values():
        v.sort(key=lambda t: t[0])
    return seqs, n_png, skipped


def decode(data: bytes, size: int):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def process_tar(path: Path, label: str, out_dir: Path, cfg: WildConfig) -> tuple[list[dict], dict]:
    """Write the clips of one tar. Returns (index rows, stats). Raises if the tar cannot be opened."""
    rows = []
    stats = dict(selected=0, read_ok=0, png=0, skipped=0)
    with tarfile.open(path, mode="r:*") as tf:
        seqs, stats["png"], stats["skipped"] = sequences(tf, label)
        for clip, frames in sorted(seqs.items()):
            pick = [frames[i][1] for i in evenly_spaced(len(frames), cfg.frames)]
            folder = out_dir / clip
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*.jpg"):
                old.unlink()
            k = 0
            for m in pick:
                stats["selected"] += 1
                f = tf.extractfile(m)
                img = decode(f.read(), cfg.size) if f is not None else None
                if img is None:
                    continue
                if cv2.imwrite(str(folder / f"f{k:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, cfg.quality]):
                    k += 1
                    stats["read_ok"] += 1
            status = "error" if k == 0 else ("short" if k < cfg.short_below else "ok")
            rows.append(dict(video=clip, label=label, n_frames=k, source_tar=tar_key(path),
                             n_source_frames=len(frames), status=status))
    return rows, stats


def read_wild_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames != WILD_FIELDS:
            raise ValueError(f"{path} has columns {r.fieldnames}, expected {WILD_FIELDS}")
        return list(r)


def append_rows(path: Path, rows: list[dict]):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WILD_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def sample_sheet(out_dir: Path, rows: list[dict], n: int = 8, seed: int = 0, size: int = 224) -> Path | None:
    """Two rows: n real clips on top, n fake clips below (middle frame, labelled)."""
    rng = random.Random(seed)
    strips = []
    for label in ("REAL", "FAKE"):
        pool = [r for r in rows if r["label"] == label and r["status"] != "error"]
        if len(pool) < n:
            return None
        tiles = []
        for r in rng.sample(pool, n):
            imgs = sorted((out_dir / r["video"]).glob("*.jpg"))
            img = cv2.resize(cv2.imread(str(imgs[len(imgs) // 2])), (size, size))
            cv2.rectangle(img, (0, 0), (size, 20), (0, 0, 0), -1)
            cv2.putText(img, f"{label} {r['video']}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(img)
        strips.append(np.hstack(tiles))
    path = out_dir / "_samples.jpg"
    cv2.imwrite(str(path), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def run_wild_gates(out_dir: Path, rows: list[dict], n_tars: int, tars_failed: list[str], selected: int,
                   read_ok: int, size: int = 224, n_decode: int = 200, seed: int = 0) -> list[dict]:
    """W1 every tar opened, W2 >= 99% of selected members read, W3 every clip has its frames on disk,
    W4 n_decode random JPEGs decode at size x size, W5 _samples.jpg exists."""
    res = []

    def add(gate, ok, detail):
        res.append(dict(gate=gate, ok=bool(ok), detail=detail))

    opened = len({r["source_tar"] for r in rows})
    add("W1", not tars_failed and opened == n_tars,
        f"{opened}/{n_tars} tars opened and yielded clips" + (f"; failed: {tars_failed[:3]}" if tars_failed else ""))
    frac = read_ok / selected if selected else 0.0
    add("W2", frac >= 0.99, f"members read {read_ok}/{selected} = {100 * frac:.2f}% (need >= 99%)")
    missing = [r["video"] for r in rows
               if r["status"] == "error" or len(list((out_dir / r["video"]).glob("*.jpg"))) != int(r["n_frames"])]
    add("W3", rows and not missing, f"{len(rows) - len(missing)}/{len(rows)} clips have n_frames JPEGs"
        + (f"; e.g. {missing[:3]}" if missing else ""))
    pool = sorted(p for r in rows for p in (out_dir / r["video"]).glob("*.jpg"))
    pick = random.Random(seed).sample(pool, min(n_decode, len(pool)))
    good = 0
    for p in pick:
        img = cv2.imread(str(p))
        good += img is not None and img.shape[:2] == (size, size)
    add("W4", pick and good == len(pick), f"{good}/{len(pick)} random JPEGs decode at {size}x{size}")
    add("W5", (out_dir / "_samples.jpg").exists(), "_samples.jpg (8 real, 8 fake clips) "
        + ("ok" if (out_dir / "_samples.jpg").exists() else "missing"))
    return res


def config_dict(cfg: WildConfig) -> dict:
    return asdict(cfg)
