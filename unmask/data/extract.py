"""Extract one DFDC part: real videos and up to N usable fakes each, face crops as JPEG (PLAN 6.2).

Per group (a real video and its fakes, fixed-seed fake order):
  * read n evenly spaced frames of the real, detect and choose the face, crop with 30% margin to 224 x 224
  * for each fake: read the same frame indices, crop with the real's boxes, score against the real crops
    (in memory, before JPEG), save and mark flagged = score < threshold
  * flagged fakes are saved but do not count toward max_fakes; at most max_tries fakes are read per real
Resumable by group via index.csv. The zip is only read.
"""
import random
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .batch import EVAL_PARTS
from .dfdc_zip import build_groups, load_metadata, read_frames
from .faces import crops_for
from .index import IndexWriter, done_groups, read_index, video_dir
from .score import THRESHOLD, local_noise_score

KB_PER_FRAME_Q95 = 12.9  # 147 KB / 16 frames at q90, about +40% at q95


@dataclass
class ExtractConfig:
    frames: int = 16
    max_fakes: int = 5
    max_tries: int = 10
    margin: float = 0.30
    size: int = 224
    quality: int = 95
    threshold: float = THRESHOLD
    seed: int = 0
    min_frames_frac: float = 0.5  # a real needs at least this share of the requested frames


def config_for_part(part: int) -> ExtractConfig:
    return ExtractConfig(frames=32 if part in EVAL_PARTS else 16)


def planned_videos(fakes_of: dict, max_fakes: int) -> int:
    return sum(1 + min(max_fakes, len(v)) for v in fakes_of.values())


def estimate_mb(n_videos: int, frames: int, flag_overhead: float = 0.10) -> float:
    return n_videos * (1 + flag_overhead) * frames * KB_PER_FRAME_Q95 / 1024


def save_crops(crops, folder: Path, quality: int):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.jpg"):
        old.unlink()
    for k, c in enumerate(crops):
        if not cv2.imwrite(str(folder / f"f{k:02d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise OSError(f"could not write {folder / f'f{k:02d}.jpg'}")


def fake_order(fakes_of: dict, seed: int) -> dict:
    rng = random.Random(seed)
    return {r: rng.sample(v, len(v)) for r, v in sorted(fakes_of.items())}


def extract_part(zip_path: Path, part: int, out_dir: Path, cfg: ExtractConfig, detector, log=None,
                 limit_groups: int = 0, reader=read_frames, show_bar: bool = True) -> dict:
    """Extract (or resume) one part. detector(frames) -> list of boxes or None. Returns stats."""
    say = log.info if log else print
    tag = f"part {part:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"
    rows_done = read_index(index_path)
    skip = done_groups(rows_done)
    stats = Counter()
    scores = []
    t0 = time.time()

    with zipfile.ZipFile(zip_path) as zf:
        meta, prefix, raw = load_metadata(zf)
        (out_dir / "metadata.json").write_bytes(raw)
        reals, fakes_of, info = build_groups(meta, set(zf.namelist()), prefix)
        order = fake_order(fakes_of, cfg.seed)
        todo = [r for r in reals if r not in skip]
        if limit_groups:
            todo = todo[:limit_groups]
        planned = planned_videos({r: fakes_of[r] for r in todo}, cfg.max_fakes)
        if skip:
            say(f"{tag}: resuming, {len(skip)} groups already done, {len(todo)} to go")
        say(f"{tag}: {len(reals)} reals, {sum(len(v) for v in fakes_of.values())} fakes with original in part; "
            f"{len(todo)} groups, about {planned} videos planned")

        with IndexWriter(index_path) as writer:
            if not rows_done:
                # reals listed in metadata but absent from the zip get a row now, so G1 can see them
                missing_rows = [dict(video=r, label="REAL", n_frames=0, status="missing", part=part)
                                for r in info["reals_missing"]]
                if missing_rows:
                    writer.write_group(missing_rows)
                    stats["real_missing"] += len(missing_rows)

            bar = tqdm(total=planned, unit="video", dynamic_ncols=True, desc=tag, disable=not show_bar)
            for gi, real in enumerate(todo, 1):
                rows = []
                want = min(cfg.max_fakes, len(order[real]))
                bar.set_description(f"{tag}: group {gi}/{len(todo)} detecting faces")
                frames, idx = reader(zf, prefix + real, cfg.frames)
                stats["videos_read"] += 1
                status = None
                if len(frames) < max(1, int(cfg.frames * cfg.min_frames_frac)):
                    status = "read_error"
                    stats["read_errors"] += 1
                else:
                    boxes = detector(frames)
                    if boxes is None:
                        status = "no_face"
                if status:
                    rows.append(dict(video=real, label="REAL", n_frames=len(frames), status=status, part=part))
                    stats[f"real_{status}"] += 1
                    bar.update(1 + want)
                else:
                    ref_shape = frames[0].shape[:2]
                    real_crops = crops_for(frames, boxes, ref_shape, cfg.margin, cfg.size)
                    save_crops(real_crops, video_dir(out_dir, real), cfg.quality)
                    rows.append(dict(video=real, label="REAL", n_frames=len(real_crops), status="ok", part=part))
                    stats["real_ok"] += 1
                    bar.update(1)

                    usable = tries = 0
                    for fake in order[real]:
                        if usable >= cfg.max_fakes or tries >= cfg.max_tries:
                            break
                        tries += 1
                        bar.set_description(f"{tag}: group {gi}/{len(todo)} cropping fakes")
                        fframes, fidx = reader(zf, prefix + fake, cfg.frames, indices=idx)
                        stats["videos_read"] += 1
                        if fidx != idx:
                            rows.append(dict(video=fake, label="FAKE", original=real, n_frames=len(fframes),
                                             status="read_error", part=part))
                            stats["read_errors"] += 1
                            stats["fake_read_error"] += 1
                            bar.total += 1  # does not count toward max_fakes
                            bar.update(1)
                            continue
                        fcrops = crops_for(fframes, boxes, ref_shape, cfg.margin, cfg.size)
                        score, noise = local_noise_score(fcrops, real_crops)
                        flagged = score < cfg.threshold
                        scores.append(score)
                        save_crops(fcrops, video_dir(out_dir, fake), cfg.quality)
                        rows.append(dict(video=fake, label="FAKE", original=real, n_frames=len(fcrops), status="ok",
                                         score=f"{score:.3f}", flagged=int(flagged), noise=f"{noise:.3f}", part=part))
                        if flagged:
                            stats["fake_flagged"] += 1
                            bar.total += 1
                        else:
                            stats["fake_usable"] += 1
                            usable += 1
                        bar.update(1)
                    if usable < want:
                        bar.total -= want - usable
                        stats["reals_short_of_fakes"] += 1
                    if order[real] and usable == 0:
                        stats["reals_without_usable_fake"] += 1
                    if not order[real]:
                        stats["reals_without_any_fake"] += 1

                writer.write_group(rows)
                n_scored = stats["fake_usable"] + stats["fake_flagged"]
                bar.set_postfix(real=stats["real_ok"], fake=stats["fake_usable"], flagged=stats["fake_flagged"],
                                flag_pct=f"{100 * stats['fake_flagged'] / max(1, n_scored):.1f}%",
                                no_face=stats["real_no_face"], err=stats["read_errors"])
            bar.close()

    stats["seconds"] = round(time.time() - t0, 1)
    stats["scores"] = scores
    stats["info"] = info
    stats["config"] = asdict(cfg)
    return stats


def score_percentiles(rows: list[dict]) -> list[float]:
    s = [float(r["score"]) for r in rows if r["label"] == "FAKE" and r["status"] == "ok" and r["score"]]
    return np.percentile(s, [0, 10, 25, 50, 75, 90]).round(2).tolist() if s else []
