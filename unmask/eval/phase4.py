"""Phase 4 (PLAN 11): score sets, calibrate on part 08, choose options on part 08, then test each set once.

Scores are stored per frame as z = fake logit - real logit, so a temperature T can be applied afterwards:
calibrated frame probability = sigmoid(z / T); video score = mean over its frames.
Sets: val08 (DFDC part 08, validation), test09 (DFDC part 09), celebdf (Celeb-DF v2 test list), wild (WildDeepfake
test). Flagged near-identical fakes are excluded on DFDC, as in training and validation.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..common import CROPS_DIR
from ..data.dataset import load_groups, load_image
from ..data.dfdc_zip import evenly_spaced
from ..data.index import read_index, video_dir
from .metrics import binary_metrics
from .views import view_fn

SETS = ("val08", "test09", "celebdf", "wild")
TEST_SETS = ("test09", "celebdf", "wild")
SET_LABELS = {"val08": "DFDC part 08 (validation)", "test09": "DFDC part 09", "celebdf": "Celeb-DF v2",
              "wild": "WildDeepfake"}


@dataclass
class Video:
    key: str
    label: int
    group: str
    frames: list = field(default_factory=list)   # JPEG paths


def _frames(d: Path, n: int) -> list[str]:
    return [str(Path(d) / f"f{i:02d}.jpg") for i in range(n)]


def load_set(name: str, crops_dir: Path = CROPS_DIR) -> list[Video]:
    """Every usable video of a set with all its stored frames. Group = bootstrap unit (PLAN 11)."""
    if name in ("val08", "test09"):
        part = 8 if name == "val08" else 9
        out = []
        for g in sorted(load_groups([part], crops_dir).values(), key=lambda g: g.key):
            out.append(Video(g.key, 0, g.key, _frames(g.real_dir, g.n_frames)))
            out += [Video(fk, 1, g.key, _frames(fd, fn)) for fd, fn, fk in g.fakes]
        return out
    if name == "celebdf":
        d = crops_dir / "celebdf_v2_test"
        rows = [r for r in read_index(d / "index.csv") if r["status"] == "ok"]
        return [Video(r["video"], int(r["label"] == "FAKE"), r["video"], _frames(video_dir(d, r["video"]), int(r["n_frames"])))
                for r in rows]
    if name == "wild":
        d = crops_dir / "wild_test"
        import csv
        with open(d / "index.csv", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
        return [Video(r["video"], int(r["label"] == "FAKE"), r["source_tar"], _frames(d / r["video"], int(r["n_frames"])))
                for r in rows]
    raise ValueError(f"unknown set {name!r}")


def subsample(videos: list[Video], n: int) -> list[Video]:
    """Keep n evenly spaced stored frames per video (the stress views use 16)."""
    return [Video(v.key, v.label, v.group, [v.frames[i] for i in evenly_spaced(len(v.frames), n)]) for v in videos]


@torch.no_grad()
def frame_logits(model, paths: list[str], device: str, transform=None, flip: bool = False, batch: int = 128,
                 threads: int = 6, progress=None) -> np.ndarray:
    """z = fake logit - real logit for every path (threaded JPEG decoding, no worker processes)."""
    model.eval()

    def load(p):
        img = load_image(p)
        return transform(img) if transform is not None else img

    chunks = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    out = []
    with ThreadPoolExecutor(threads) as pool:
        nxt = pool.map(load, chunks[0]) if chunks else None
        for c, _ in enumerate(chunks):
            imgs = list(nxt)
            if c + 1 < len(chunks):
                nxt = pool.map(load, chunks[c + 1])
            x = torch.stack(imgs).to(device, non_blocking=True).float().div_(255)
            if flip:
                x = torch.flip(x, dims=[3])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, _ = model(x)
            logits = logits.float()
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            if progress is not None:
                progress.update(1)
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@dataclass
class Scores:
    keys: list
    labels: np.ndarray
    groups: list
    offsets: np.ndarray        # video i owns z[offsets[i]:offsets[i+1]]
    z: np.ndarray

    def video_probs(self, temperature: float = 1.0) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.clip(self.z / temperature, -60, 60)))
        return np.add.reduceat(p, self.offsets[:-1]) / np.diff(self.offsets)

    def save(self, path: Path, **meta):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(self.keys), labels=self.labels, groups=np.array(self.groups),
                            offsets=self.offsets, z=self.z, meta=np.array([repr(meta)]))

    @classmethod
    def load(cls, path: Path) -> "Scores":
        d = np.load(path, allow_pickle=False)
        return cls(d["keys"].tolist(), d["labels"], d["groups"].tolist(), d["offsets"], d["z"])


def score_videos(model, videos: list[Video], device: str, transform=None, flip: bool = False, progress=None,
                 batch: int = 128) -> Scores:
    paths = [p for v in videos for p in v.frames]
    z = frame_logits(model, paths, device, transform=transform, flip=flip, progress=progress, batch=batch)
    offsets = np.concatenate([[0], np.cumsum([len(v.frames) for v in videos])]).astype(np.int64)
    return Scores([v.key for v in videos], np.array([v.label for v in videos]), [v.group for v in videos], offsets, z)


def view_transform(view: str):
    return None if view == "clean" else view_fn(view)


# ---------------------------------------------------------------- calibration and choices
def fit_temperature(scores: Scores) -> float:
    """T minimising video-level log loss on validation (grid 0.25-8, log spaced)."""
    best_t, best = 1.0, np.inf
    y = scores.labels
    for t in np.exp(np.linspace(np.log(0.25), np.log(8.0), 121)):
        p = np.clip(scores.video_probs(t), 1e-6, 1 - 1e-6)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if loss < best:
            best_t, best = float(t), loss
    return best_t


def choose_band(probs: np.ndarray, labels: np.ndarray, max_err: float = 0.05,
                fallback: tuple = (0.35, 0.65)) -> dict:
    """Two thresholds with per-class error control, so they do not depend on the share of fakes in the data
    (part 08 is about 76% fake; a real-world upload is not):
      high = smallest score that at most max_err of REAL videos reach   -> 'Likely Fake' at or above it
      low  = largest score that at most max_err of FAKE videos stay at  -> 'Likely Real' at or below it
    Videos between them are 'Inconclusive'. If the classes are well separated (the 95th percentile of reals lies
    below the 5th percentile of fakes), the gap between those two points becomes the Inconclusive zone, so a
    video unlike both validation classes is not forced into a verdict; both error limits still hold.
    Fallback 0.35 / 0.65 when a class has too few videos."""
    s, y = np.asarray(probs, float), np.asarray(labels, int)
    real, fake = s[y == 0], s[y == 1]
    if len(real) < 20 or len(fake) < 20:
        return dict(low=fallback[0], high=fallback[1], fitted=False)
    q_real = float(np.quantile(real, 1 - max_err, method="higher"))
    q_fake = float(np.quantile(fake, max_err, method="lower"))
    low, high = min(q_real, q_fake), max(q_real, q_fake)
    return dict(low=low, high=high, fitted=True, real_above_high=float(np.mean(real > high)),
                fake_below_low=float(np.mean(fake < low)))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, int)
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & ((conf < b) if b < 1.0 else (conf <= b))
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def bootstrap_auc(labels, scores, groups, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the AUC, resampling whole groups with replacement (PLAN 11)."""
    y, s = np.asarray(labels), np.asarray(scores)
    uniq = sorted(set(groups))
    idx_of = {g: [] for g in uniq}
    for i, g in enumerate(groups):
        idx_of[g].append(i)
    idx_lists = [np.array(idx_of[g]) for g in uniq]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_lists[k] for k in pick])
        if len(set(y[ii].tolist())) == 2:
            aucs.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def full_metrics(scores: Scores, temperature: float, band: dict, n_boot: int = 1000) -> dict:
    p = scores.video_probs(temperature)
    y = scores.labels
    m = binary_metrics(y, p)
    lo, hi = bootstrap_auc(y, p, scores.groups, n=n_boot)
    inside = (p > band["low"]) & (p < band["high"])
    decided = ~inside
    acc_out = float(((p[decided] >= band["high"]).astype(int) == y[decided]).mean()) if decided.any() else float("nan")
    bal = 0.5 * (np.mean(p[y == 1] >= 0.5) + np.mean(p[y == 0] < 0.5)) if len(set(y.tolist())) == 2 else float("nan")
    return m | dict(auc_lo=lo, auc_hi=hi, ece=ece(p, y), balanced_acc=float(bal), inconclusive=float(inside.mean()),
                    acc_outside_band=acc_out)


def val_robust(view_probs: dict, labels: np.ndarray) -> float:
    """Mean AUC over the clean, tight and degraded views (video probabilities per view)."""
    return float(np.mean([roc_auc_score(labels, view_probs[v]) for v in ("clean", "tight", "degraded")]))
