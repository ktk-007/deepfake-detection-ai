"""Training-loop building blocks (PLAN 9): schedule, validation, one epoch, resumable state, early stopping.

Kept free of argparse and printing so the pieces are unit-testable; scripts/train.py wires them together.
"""
import csv
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import load_image
from ..data.dfdc_zip import evenly_spaced
from ..eval.metrics import binary_metrics, video_scores
from ..models.build import frame_probs
from ..models.losses import LossConfig, gend_loss


# ---------------------------------------------------------------- schedule
def warmup_cosine(total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6):
    """LambdaLR factor: linear warm-up from ~0 to base_lr, then cosine down to min_lr at total_steps."""
    floor = min_lr / base_lr

    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(1.0, max(0.0, t))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return f


# ---------------------------------------------------------------- validation
def eval_items(groups, frames_per_video: int = 0) -> list[tuple[str, int, str]]:
    """(path, label, video key) for the real and every usable fake of each group.
    frames_per_video > 0 keeps that many evenly spaced stored frames per video; 0 keeps all."""
    out = []

    def add(d, n, label, key):
        idx = range(n) if frames_per_video <= 0 else evenly_spaced(n, frames_per_video)
        out.extend((str(Path(d) / f"f{i:02d}.jpg"), label, key) for i in idx)

    for g in sorted(groups, key=lambda g: g.key):
        add(g.real_dir, g.n_frames, 0, g.key)
        for fdir, fn, fkey in g.fakes:
            add(fdir, fn, 1, fkey)
    return out


@torch.no_grad()
def predict_items(model, items, device, batch_size: int = 128, threads: int = 6, progress=None,
                  autocast: bool = True, transform=None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Frame fake-probabilities for items, decoding JPEGs in a thread pool (no extra worker processes,
    so no second set of DataLoader workers next to the training ones; WinError 1455).
    transform: optional deterministic uint8 CHW -> uint8 CHW function (the part-08 stress views)."""
    model.eval()
    probs, labels, keys = [], [], []
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def load(it):
        img = load_image(it[0])
        return transform(img) if transform is not None else img

    with ThreadPoolExecutor(threads) as pool:
        nxt = pool.map(load, batches[0]) if batches else None
        for b, chunk in enumerate(batches):
            imgs = list(nxt)
            if b + 1 < len(batches):
                nxt = pool.map(load, batches[b + 1])  # prefetch the next batch
            x = torch.stack(imgs).to(device, non_blocking=True).float().div_(255)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast and device == "cuda"):
                logits, _ = model(x)
            probs.append(frame_probs(logits).cpu())
            labels += [it[1] for it in chunk]
            keys += [it[2] for it in chunk]
            if progress is not None:
                progress.update(1)
    if not probs:
        return np.zeros(0), np.zeros(0, dtype=int), []
    return torch.cat(probs).numpy(), np.array(labels), keys


def validate(model, items, device, **kw) -> tuple[dict, tuple]:
    """Video-level metrics (score = mean frame probability). Returns (metrics, (videos, labels, scores))."""
    p, y, k = predict_items(model, items, device, **kw)
    vids, vy, vs = video_scores(p, y, k)
    return binary_metrics(vy, vs), (vids, vy, vs)


# ---------------------------------------------------------------- training
def loss_for(kind: str, label_smoothing: float) -> LossConfig:
    """'gend' = CE + 0.1 align + 0.5 unif (GenD); 'ce' = cross-entropy only (E4d, EfficientNet)."""
    if kind == "gend":
        return LossConfig(label_smoothing=label_smoothing)
    if kind == "ce":
        return LossConfig(label_smoothing=label_smoothing, w_align=0.0, w_unif=0.0)
    raise ValueError(f"unknown loss {kind!r}")


def compute_loss(logits, z, y, cfg: LossConfig):
    if z is None:  # models without an embedding (EfficientNet) can only use CE
        cfg = LossConfig(ce=cfg.ce, label_smoothing=cfg.label_smoothing, w_align=0.0, w_unif=0.0)
        z = logits
    return gend_loss(logits, z, y, cfg)


def train_epoch(model, loader, opt, sched, params, loss_cfg: LossConfig, device: str, clip: float = 1.0,
                bar=None, max_steps: int = 0) -> dict:
    """One pass over the loader. Returns mean loss parts, images, seconds. Raises on a non-finite loss."""
    model.train()
    sums = dict(loss=0.0, ce=0.0, align=0.0, unif=0.0)
    n_steps = n_img = 0
    t0 = time.time()
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, z = model(x)
        loss, parts = compute_loss(logits, z, y, loss_cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {n_steps}: {parts}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        sums["loss"] += loss.item()
        for k in ("ce", "align", "unif"):
            sums[k] += parts[k]
        n_steps += 1
        n_img += x.size(0)
        if bar is not None:
            bar.update(1)
            bar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{parts['ce']:.3f}",
                            img_s=f"{n_img / max(1e-9, time.time() - t0):.0f}",
                            lr=f"{sched.get_last_lr()[0]:.1e}")
        if max_steps and n_steps >= max_steps:
            break
    sec = time.time() - t0
    out = {k: v / max(1, n_steps) for k, v in sums.items()}
    return out | dict(steps=n_steps, images=n_img, seconds=sec, img_s=n_img / max(1e-9, sec))


# ---------------------------------------------------------------- early stopping
class EarlyStop:
    """Best checkpoint by raw AUC. Stop after `patience` epochs with neither a new raw best nor a new best
    mean AUC over the last `smooth` epochs (PLAN 9.4, smoothing from the risk table in section 17)."""

    def __init__(self, patience: int = 5, smooth: int = 3):
        self.patience, self.smooth = patience, smooth
        self.history: list[float] = []
        self.best_raw = -math.inf
        self.best_epoch = -1
        self.best_smooth = -math.inf
        self.bad = 0

    def update(self, epoch: int, auc: float) -> tuple[bool, bool]:
        """Returns (is_new_best, should_stop)."""
        auc = -math.inf if auc is None or math.isnan(auc) else auc
        self.history.append(auc)
        new_best = auc > self.best_raw
        if new_best:
            self.best_raw, self.best_epoch = auc, epoch
        sm = float(np.mean(self.history[-self.smooth:]))
        better_smooth = sm > self.best_smooth + 1e-6
        if better_smooth:
            self.best_smooth = sm
        # Progress = a new raw best or a new smoothed best; never stop on the epoch that set a new best.
        self.bad = 0 if (new_best or better_smooth) else self.bad + 1
        return new_best, self.patience > 0 and self.bad >= self.patience

    def state_dict(self) -> dict:
        return dict(self.__dict__)

    def load_state_dict(self, d: dict):
        self.__dict__.update(d)


# ---------------------------------------------------------------- state and checkpoints
def rng_state() -> dict:
    st = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_last(path: Path, model, opt, sched, epoch: int, stopper: EarlyStop, config: dict, history: list):
    """Everything needed to resume after `epoch` (0-based, completed). Written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(kind="last", config=config, arch=model.arch, linear_probe=model.linear_probe,
                    trainable=model.trainable_state_dict(), optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(), epoch=epoch, stopper=stopper.state_dict(),
                    history=history, rng=rng_state()), tmp)
    tmp.replace(path)


def save_best(path: Path, model, epoch: int, config: dict, metrics: dict):
    """Small inference checkpoint: trainable tensors + config + the validation metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(kind="best", config=config, arch=model.arch, linear_probe=model.linear_probe,
                    trainable=model.trainable_state_dict(), epoch=epoch, val=metrics), tmp)
    tmp.replace(path)


def save_epoch(path: Path, model, epoch: int, config: dict, metrics: dict):
    """Per-epoch trainable tensors (CLIP: about 0.2 MB) for epoch averaging (PLAN 9.6)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(dict(kind="epoch", config=config, arch=model.arch, linear_probe=model.linear_probe,
                    trainable=model.trainable_state_dict(), epoch=epoch, val=metrics), tmp)
    tmp.replace(path)


def load_model_checkpoint(path: Path, device: str = "cpu"):
    """Rebuild a model from best.pt / avg.pt / ep_NN.pt / last.pt. Frozen CLIP weights come from open_clip's
    cache; EfficientNet and the test model carry every tensor, so no download is needed for them."""
    from ..models.build import build_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    model = build_model(arch, linear_probe=ck.get("linear_probe", False), grad_ckpt=False,
                        pretrained=arch in ("vit_b16", "vit_l14"))
    load_trainable(model, ck["trainable"])
    return model.to(device).eval(), ck


def load_trainable(model, state: dict):
    """Load trainable tensors (strict about what must be there)."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise KeyError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    need = {n for n, p in model.named_parameters() if p.requires_grad}
    lacking = need - set(state)
    if lacking:
        raise KeyError(f"trainable parameters missing from checkpoint: {sorted(lacking)[:5]}")


# ---------------------------------------------------------------- run records
METRIC_FIELDS = ["epoch", "loss", "ce", "align", "unif", "val_auc", "val_ap", "val_eer", "val_logloss",
                 "val_acc", "val_auc_smooth", "lr_end", "img_s", "train_s", "val_s", "best", "checkpoint"]

RESULT_FIELDS = ["finished", "name", "arch", "linear_probe", "loss", "aug", "paired", "include_flagged",
                 "train_parts", "train_reals", "val_part", "val_videos", "val_frames", "batch", "lr", "seed",
                 "epochs_run", "best_epoch", "val_auc", "val_ap", "val_eer", "val_logloss", "img_s",
                 "peak_vram_gb", "minutes", "note",
                 # filled by scripts\eval_val.py and scripts\average_ckpt.py (PLAN 9.5, 9.6)
                 "val_clean", "val_tight", "val_degraded", "val_robust", "avg_epochs", "avg_val_robust",
                 "avg_used"]


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    if not path.exists():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write_rows(path: Path, fields: list[str], rows: list[dict]):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def _widen(path: Path, fields: list[str]) -> list[str]:
    """Rewrite the file with `fields` (plus any extra old columns at the end) if its header differs.
    Existing rows are kept unchanged; new columns are empty."""
    old, rows = _read_rows(path)
    if not old or old == fields:
        return fields
    full = list(fields) + [c for c in old if c not in fields]
    if full != old:
        _write_rows(path, full, rows)
    return full


def append_csv(path: Path, fields: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        fields = _widen(path, fields)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: _fmt(row.get(k, "")) for k in fields})


def update_result(path: Path, name: str, values: dict, fields: list[str] = RESULT_FIELDS) -> bool:
    """Set `values` on the last row of run `name` (widening the header if needed). False if no such row."""
    if not path.exists():
        return False
    fields = _widen(path, fields)
    _, rows = _read_rows(path)
    idx = [i for i, r in enumerate(rows) if r.get("name") == name]
    if not idx:
        return False
    rows[idx[-1]].update({k: _fmt(v) for k, v in values.items()})
    _write_rows(path, fields, rows)
    return True


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.5f}"
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return v


def write_val_scores(path: Path, vids, vy, vs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "label", "score"])
        for v, y, s in zip(vids, vy, vs):
            w.writerow([v, int(y), f"{float(s):.6f}"])


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
