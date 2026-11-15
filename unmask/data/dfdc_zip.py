"""DFDC part zips: find them, read metadata.json, build real -> fakes groups, stream single videos.

A zip is never unpacked. Each video is copied to its own temp file, decoded and the temp file deleted.
"""
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ZIP_RE = re.compile(r"^dfdc_train_part_(\d+)\.zip$")


def find_part_zips(zip_dir: Path) -> dict[int, Path]:
    """{part number: path} for every dfdc_train_part_NN.zip in zip_dir."""
    out = {}
    if zip_dir.is_dir():
        for p in sorted(zip_dir.iterdir()):
            m = ZIP_RE.match(p.name)
            if m and p.is_file():
                out[int(m.group(1))] = p
    return out


def load_metadata(zf: zipfile.ZipFile) -> tuple[dict, str, bytes]:
    """(metadata dict, member prefix, raw bytes). Raises if no metadata.json."""
    meta_name = next((n for n in zf.namelist() if n.endswith("metadata.json")), None)
    if meta_name is None:
        raise ValueError("no metadata.json in zip")
    raw = zf.read(meta_name)
    return json.loads(raw), meta_name[: -len("metadata.json")], raw


def build_groups(meta: dict, members: set[str], prefix: str = ""):
    """Group videos by their original real video.

    Returns (reals, fakes_of, info):
      reals     sorted real video names present in the zip
      fakes_of  {real: sorted fake names present in the zip whose "original" is that real}
      info      counts: n_real_meta, n_fake_meta, reals_missing (list), fakes_orphan, fakes_missing
    """
    real_meta = sorted(k for k, v in meta.items() if v.get("label") == "REAL")
    fake_meta = sorted(k for k, v in meta.items() if v.get("label") == "FAKE")
    reals = [r for r in real_meta if prefix + r in members]
    real_set = set(reals)
    fakes_of: dict[str, list[str]] = {r: [] for r in reals}
    orphan = missing = 0
    for f in fake_meta:
        if prefix + f not in members:
            missing += 1
            continue
        orig = meta[f].get("original")
        if orig in real_set:
            fakes_of[orig].append(f)
        else:
            orphan += 1
    info = dict(
        n_real_meta=len(real_meta),
        n_fake_meta=len(fake_meta),
        reals_missing=[r for r in real_meta if r not in real_set],
        fakes_orphan=orphan,
        fakes_missing=missing,
    )
    return reals, fakes_of, info


def evenly_spaced(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, n).round().astype(int).tolist()))


def read_frames(zf: zipfile.ZipFile, member: str, n_frames: int, indices=None):
    """Stream one video to a temp file and return (frames_bgr, frame_indices).

    indices=None picks n_frames evenly spaced frames. The temp file is always deleted.
    Returns ([], []) when the video cannot be decoded.
    """
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="unmask_")
    os.close(fd)
    try:
        with zf.open(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        return read_frames_file(tmp, n_frames, indices)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def read_frames_file(path: str, n_frames: int, indices=None):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return [], []
        if indices is None:
            indices = evenly_spaced(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n_frames)
        if not indices:
            return [], []
        wanted, frames, got = set(indices), [], []
        last = max(indices)
        i = 0
        while i <= last:
            if not cap.grab():
                break
            if i in wanted:
                ok, f = cap.retrieve()
                if ok and f is not None:
                    frames.append(f)
                    got.append(i)
            i += 1
        return frames, got
    finally:
        cap.release()
