"""End-to-end smoke test on the batch 1 crops (PLAN 7.3), about 3 to 5 minutes.

CLIP ViT-B/16 with LayerNorm tuning: paired sampler + temp grouped split, about 150 steps, then checks
  1. every loss is finite          2. the loss falls (mean of the last 20 steps < mean of the first 20)
  3. video-level validation AUC is computed
  4. a checkpoint saves, reloads into a fresh model and gives identical predictions
On success writes data\\pipeline_ok.json and deletes the zips of parts that passed their gates.
On failure keeps everything and prints the failing check. The first run downloads the OpenAI CLIP
ViT-B/16 weights (about 0.6 GB, pre-approved).

  python scripts\\smoke_train.py
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unmask.common import PIPELINE_OK, PROJECT_ROOT, RUNS_DIR, SPLITS_DIR, setup_logger  # noqa: E402
from unmask.data.augment import build_augment  # noqa: E402
from unmask.data.batch import delete_done_zips, done_parts, free_gb, next_message  # noqa: E402
from unmask.data.dataset import (EvalFrames, ItemDataset, PairedBatchSampler, collate,  # noqa: E402
                                 load_groups)
from unmask.data.splits import grouped_split, load_split, save_split  # noqa: E402
from unmask.eval.metrics import binary_metrics, video_scores  # noqa: E402
from unmask.models.clip_ln import ClipLN  # noqa: E402
from unmask.models.losses import LossConfig, gend_loss  # noqa: E402
from unmask.train.checkpoint import load_clip_checkpoint, save_clip_checkpoint  # noqa: E402

SMOKE_PARTS = (0, 1)


@torch.no_grad()
def predict(model, loader, device, autocast=True):
    model.eval()
    probs, labels, keys = [], [], []
    for x, y, k in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits, _ = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1)[:, 1].cpu())
        labels.append(y)
        keys += k
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--pairs-per-batch", type=int, default=8, help="8 pairs x 4 frames x 2 = batch 64")
    ap.add_argument("--frames-per-video", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log, log_path = setup_logger("smoke_train")
    t0 = time.time()
    checks = {}

    def finish(ok: bool):
        log.info("")
        log.info("==== SMOKE TEST SUMMARY ====")
        for name, (passed, detail) in checks.items():
            log.info(f"{name}: {'PASS' if passed else 'FAIL'} - {detail}")
        if ok:
            deleted, status = delete_done_zips(log)
            log.info("pipeline_ok.json written")
            log.info(f"zips deleted: {status}")
            log.info(f"free disk: {free_gb(PROJECT_ROOT):.1f} GB | time {(time.time() - t0) / 60:.1f} min")
            log.info("NEXT: " + next_message(done_parts()))
        else:
            log.info("RESULT: FAILED. Nothing deleted, zips kept. Paste this summary back.")
        log.info(f"log: {log_path}")
        log.info("============================")

    done = done_parts()
    missing = [p for p in SMOKE_PARTS if p not in done]
    if missing:
        checks["crops"] = (False, f"parts without DONE.json: {missing}. Run scripts\\process_batch.py first.")
        return finish(False)
    if not torch.cuda.is_available():
        checks["cuda"] = (False, "CUDA not available")
        return finish(False)
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------------------------------------------- data and temp split
    groups = load_groups(SMOKE_PARTS)
    split_path = SPLITS_DIR / "temp_val.json"
    if not split_path.exists():
        tr, va = grouped_split(list(groups), 0.2, seed=0)
        save_split(split_path, tr, va, parts=list(SMOKE_PARTS), seed=0, val_frac=0.2,
                   note="temporary validation until parts 48 and 49 are extracted")
        log.info(f"temp split written: {split_path}")
    split = load_split(split_path)
    train_g = [groups[k] for k in split["train"] if k in groups]
    val_g = [groups[k] for k in split["val"] if k in groups]
    log.info(f"groups: train {len(train_g)}, val {len(val_g)} (grouped by original video) | "
             f"val videos: {len(val_g)} real, {sum(len(g.fakes) for g in val_g)} fake")

    sampler = PairedBatchSampler(train_g, args.pairs_per_batch, args.frames_per_video, paired=True, seed=args.seed)
    bs = args.pairs_per_batch * args.frames_per_video * 2
    kw = dict(num_workers=args.workers, pin_memory=True, collate_fn=collate,
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(ItemDataset(build_augment("light")), batch_sampler=sampler, **kw)
    val_dl = DataLoader(EvalFrames(val_g), batch_size=128, shuffle=False, **kw)

    # ---------------------------------------------------------------- model
    log.info("loading CLIP ViT-B/16 (OpenAI weights; the first run downloads about 0.6 GB)")
    model = ClipLN("vit_b16", pretrained="openai").to(device)
    total, trainable = model.param_counts()
    log.info(f"parameters: total {total:,}, trainable {trainable:,} ({100 * trainable / total:.3f}%)")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    loss_cfg = LossConfig()

    # ---------------------------------------------------------------- train
    torch.cuda.reset_peak_memory_stats()
    losses, step, epoch = [], 0, 0
    bar = tqdm(total=args.steps, unit="step", dynamic_ncols=True, desc="smoke: training")
    t_train = time.time()
    n_img = 0
    finite = True
    while step < args.steps and finite:
        sampler.set_epoch(epoch)
        model.train()
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, z = model(x)
            loss, parts = gend_loss(logits, z, y, loss_cfg)
            if not torch.isfinite(loss):
                finite = False
                log.info(f"non-finite loss at step {step}: {parts}")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            n_img += x.size(0)
            step += 1
            bar.set_description(f"smoke: epoch {epoch} step {step}/{args.steps}")
            bar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{parts['ce']:.3f}", align=f"{parts['align']:.3f}",
                            unif=f"{parts['unif']:.3f}", img_s=f"{n_img / (time.time() - t_train):.0f}")
            bar.update(1)
            if step >= args.steps:
                break
        epoch += 1
    bar.close()
    train_s = time.time() - t_train
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    log.info(f"trained {step} steps ({epoch} sampler epochs) in {train_s:.0f} s, {n_img / train_s:.1f} img/s, "
             f"batch {bs}, peak VRAM {peak_gb:.2f} GB")

    checks["1 loss finite"] = (finite and len(losses) == args.steps, f"{len(losses)} finite steps")
    k = min(20, len(losses) // 2)
    first, last = (float(np.mean(losses[:k])), float(np.mean(losses[-k:]))) if k else (math.nan, math.nan)
    checks["2 loss falling"] = (k > 0 and last < first, f"mean first {k} steps {first:.4f} -> last {k} {last:.4f}")
    if not all(c[0] for c in checks.values()):
        return finish(False)

    # ---------------------------------------------------------------- validate
    log.info(f"validating on {len(val_dl.dataset)} frames")
    probs, labels, keys = predict(model, tqdm(val_dl, desc="smoke: validation", unit="batch", dynamic_ncols=True),
                                  device)
    _, vy, vs = video_scores(probs, labels, keys)
    m = binary_metrics(vy, vs)
    auc_ok = not math.isnan(m.get("auc", math.nan))
    checks["3 validation AUC"] = (auc_ok, f"video AUC {m.get('auc', math.nan):.4f}, AP {m.get('ap', math.nan):.4f}, "
                                          f"{m['n_real']} real / {m['n_fake']} fake videos")
    if auc_ok and m["auc"] <= 0.5:
        log.info("note: AUC is not above 0.5 after this short run; Gate 2 requires a clear margin on a real run.")

    # ---------------------------------------------------------------- checkpoint round trip
    ck_path = RUNS_DIR / "smoke" / "last.pt"
    cfg = dict(arch="vit_b16", steps=args.steps, batch=bs, lr=args.lr, loss=loss_cfg.__dict__, aug="light",
               split=str(split_path), seed=args.seed)
    save_clip_checkpoint(model, ck_path, cfg)
    size_mb = ck_path.stat().st_size / 2**20
    model2, _ = load_clip_checkpoint(ck_path, device=device)
    sub = torch.utils.data.Subset(val_dl.dataset, list(range(min(64, len(val_dl.dataset)))))
    sub_dl = DataLoader(sub, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    p1, _, _ = predict(model, sub_dl, device, autocast=False)
    p2, _, _ = predict(model2, sub_dl, device, autocast=False)
    diff = float(np.abs(p1 - p2).max())
    checks["4 checkpoint reload"] = (diff < 1e-5, f"{ck_path} ({size_mb:.1f} MB), max prob diff {diff:.2e}")
    del model2

    ok = all(c[0] for c in checks.values())
    if ok:
        PIPELINE_OK.write_text(json.dumps(dict(
            passed=datetime.now().isoformat(timespec="seconds"), steps=step, batch=bs,
            img_per_s=round(n_img / train_s, 1), peak_vram_gb=round(peak_gb, 2),
            loss_first=first, loss_last=last, val=m, checkpoint=str(ck_path)), indent=1))
    finish(ok)


if __name__ == "__main__":
    main()
