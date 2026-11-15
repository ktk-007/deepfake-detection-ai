"""Extraction gates G1-G7 for one part (PLAN 7.4) and the _samples.jpg sheet."""
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .index import video_dir


def make_sample_sheet(out_dir: Path, rows: list[dict], n: int = 3, seed: int = 0, size: int = 224):
    """3 reals on top, one paired unflagged fake under each (middle frame), labelled. Returns path or None."""
    reals = [r for r in rows if r["label"] == "REAL" and r["status"] == "ok"]
    fake_of = {}
    for r in rows:
        if r["label"] == "FAKE" and r["status"] == "ok" and str(r["flagged"]) == "0":
            fake_of.setdefault(r["original"], r)
    pairs = [(r, fake_of[r["video"]]) for r in reals if r["video"] in fake_of]
    if not pairs:
        return None
    pairs = random.Random(seed).sample(pairs, min(n, len(pairs)))

    def tile(row, text):
        imgs = sorted(video_dir(out_dir, row["video"]).glob("*.jpg"))
        img = cv2.imread(str(imgs[len(imgs) // 2]))
        img = cv2.resize(img, (size, size))
        cv2.rectangle(img, (0, 0), (size, 20), (0, 0, 0), -1)
        cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    top = np.hstack([tile(r, f"REAL {Path(r['video']).stem}") for r, _ in pairs])
    bot = np.hstack([tile(f, f"FAKE {Path(f['video']).stem} s={float(f['score']):.1f}") for _, f in pairs])
    path = out_dir / "_samples.jpg"
    cv2.imwrite(str(path), np.vstack([top, bot]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def run_gates(out_dir: Path, rows: list[dict], size: int = 224, n_decode: int = 200, seed: int = 0) -> list[dict]:
    """Return [{gate, ok, warn, detail}] for G1-G7. Reads metadata.json from out_dir."""
    res = []

    def add(gate, ok, detail, warn=False):
        res.append(dict(gate=gate, ok=bool(ok), warn=bool(warn), detail=detail))

    meta_path = out_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta_reals = {k for k, v in meta.items() if v.get("label") == "REAL"}
    real_rows = {r["video"]: r for r in rows if r["label"] == "REAL"}

    missing = sorted(meta_reals - set(real_rows))
    add("G1", meta_reals and not missing, f"{len(meta_reals) - len(missing)}/{len(meta_reals)} reals have a row"
        + (f"; missing e.g. {missing[:3]}" if missing else ""))

    n_ok = sum(1 for r in real_rows.values() if r["status"] == "ok")
    share = n_ok / max(1, len(real_rows))
    add("G2", share >= 0.95, f"reals ok {n_ok}/{len(real_rows)} = {100 * share:.1f}% (need >= 95%)")

    scored = [r for r in rows if r["label"] == "FAKE" and r["status"] == "ok"]
    n_flag = sum(1 for r in scored if str(r["flagged"]) == "1")
    fshare = n_flag / max(1, len(scored))
    add("G3", fshare <= 0.20, f"flagged {n_flag}/{len(scored)} = {100 * fshare:.1f}% (max 20%, warn > 10%)",
        warn=fshare > 0.10)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    bad = []
    all_jpgs = []
    for r in ok_rows:
        jpgs = list(video_dir(out_dir, r["video"]).glob("*.jpg"))
        all_jpgs.extend(jpgs)
        if len(jpgs) != int(r["n_frames"]):
            bad.append(r["video"])
    add("G4", ok_rows and not bad, f"{len(ok_rows) - len(bad)}/{len(ok_rows)} ok videos have n_frames JPEGs"
        + (f"; wrong e.g. {bad[:3]}" if bad else ""))

    pick = random.Random(seed).sample(all_jpgs, min(n_decode, len(all_jpgs)))
    bad_img = []
    for p in pick:
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (size, size):
            bad_img.append(p.name)
    add("G5", pick and not bad_img, f"{len(pick) - len(bad_img)}/{len(pick)} random JPEGs decode at {size}x{size}")

    has_meta, has_samples = meta_path.exists(), (out_dir / "_samples.jpg").exists()
    add("G6", has_meta and has_samples, f"metadata.json {'ok' if has_meta else 'MISSING'}, "
        f"_samples.jpg {'ok' if has_samples else 'MISSING'}")

    attempted = [r for r in rows if r["status"] != "missing"]
    n_err = sum(1 for r in attempted if r["status"] == "read_error")
    eshare = n_err / max(1, len(attempted))
    add("G7", eshare < 0.01, f"read errors {n_err}/{len(attempted)} = {100 * eshare:.2f}% (need < 1%)")
    return res


def gates_pass(results: list[dict]) -> bool:
    return all(g["ok"] for g in results)
