"""Score a checkpoint on the clean, tight and degraded views of part 08 (PLAN 9.5).

val_robust = mean of the three video AUCs. Every choice between runs uses it. 16 evenly spaced frames per
video, flagged fakes excluded, threaded JPEG decoding (never a second set of DataLoader workers).
"""
import math
import time
from pathlib import Path

from ..common import CROPS_DIR
from ..data.batch import VAL_PART
from ..data.dataset import load_groups
from ..train.loop import eval_items, load_model_checkpoint, validate
from .views import VIEWS, view_fn

EVAL_FRAMES = 16


def score_views(model, val_groups, device: str, frames: int = EVAL_FRAMES, batch_size: int = 128,
                progress=None) -> dict:
    """{view: metrics} plus 'val_robust'. progress(view, n_batches) -> a tqdm-like object or None."""
    items = eval_items(val_groups, frames)
    out = {}
    for v in VIEWS:
        bar = progress(v, math.ceil(len(items) / batch_size)) if progress else None
        t0 = time.time()
        m, _ = validate(model, items, device, batch_size=batch_size, progress=bar, transform=view_fn(v))
        if bar is not None:
            bar.close()
        out[v] = m | dict(seconds=round(time.time() - t0, 1))
    aucs = [out[v]["auc"] for v in VIEWS]
    out["val_robust"] = float(sum(aucs) / len(aucs))
    out["frames_per_video"] = frames
    out["n_frames"] = len(items)
    return out


def evaluate_checkpoint(ckpt: Path, device: str, crops_dir: Path = CROPS_DIR, frames: int = EVAL_FRAMES,
                        batch_size: int = 128, progress=None) -> dict:
    model, ck = load_model_checkpoint(ckpt, device)
    val_groups = list(load_groups([VAL_PART], crops_dir, include_flagged=False).values())
    res = score_views(model, val_groups, device, frames, batch_size, progress)
    res["checkpoint"] = str(ckpt)
    res["epoch"] = ck.get("epoch")
    res["epochs"] = ck.get("epochs")
    del model
    return res
