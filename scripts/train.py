"""Train one model on DFDC crops, validating on part 08 every epoch (PLAN 9).

Training parts: 00-07 and 10-19 (default: every extracted one; --parts dev = 00-05, the Stage A development
set). Validation: part 08 only. Part 09 and the outside test sets are never loaded here. The app-test hold-out
group (data\\app_test\\app_holdout.json) is excluded from training. Settings: built-in defaults < per-arch
defaults < --config YAML < command-line flags. Epochs and patience per stage are set in the configs.

  python scripts\\train.py --name e2_b16 --config configs\\e2_b16_ln.yaml
  python scripts\\train.py --name a1_light --config configs\\a1_light.yaml --parts dev
  python scripts\\train.py --name e2_b16 --resume                     (continue after an interruption)
  python scripts\\train.py --bench 100 --arch vit_l14                 (speed and VRAM check, no run saved)

Writes runs\\<name>\\: config.json, metrics.csv, last.pt (resume state), best.pt (best part-08 AUC),
val_best.csv (part-08 video scores of the best epoch), and appends one row to results\\results.csv.
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unmask.common import CROPS_DIR, PROJECT_ROOT, RUNS_DIR, setup_logger  # noqa: E402
from unmask.data.augment import PRESETS, build_augment  # noqa: E402
from unmask.data.batch import (DEV_PARTS, VAL_PART, check_split_parts, done_parts, free_gb,  # noqa: E402
                               train_parts)
from unmask.data.dataset import ItemDataset, PairedBatchSampler, collate, load_groups  # noqa: E402
from unmask.data.splits import group_key  # noqa: E402
from unmask.models.build import ARCH_DEFAULTS, build_model, check_quickgelu  # noqa: E402
from unmask.train.loop import (METRIC_FIELDS, RESULT_FIELDS, EarlyStop, append_csv, dump_json,  # noqa: E402
                               eval_items, load_trainable, loss_for, save_best, save_epoch, save_last,
                               seed_everything,
                               set_rng_state, train_epoch, validate, warmup_cosine, write_val_scores)

RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
BENCH_CSV = PROJECT_ROOT / "results" / "bench.csv"
VRAM_LIMIT_GB = 5.2
HOLDOUT_JSON = PROJECT_ROOT / "data" / "app_test" / "app_holdout.json"


def disk_free_gb() -> float:
    """Free disk on the project drive (a function so tests can replace it)."""
    return free_gb(PROJECT_ROOT)


def holdout_keys(path: Path) -> set[str]:
    """Group keys (part/real video) kept out of training so the app can be checked on unseen videos.

    The file holds {"groups": [{"part": 6, "real": "x.mp4", ...}, ...]}; the older single-group form
    {"part": 6, "real": "x.mp4", ...} is still read. Each group is the real video and all its fakes."""
    if not path or not Path(path).exists():
        return set()
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = d["groups"] if isinstance(d, dict) and "groups" in d else d if isinstance(d, list) else [d]
    return {group_key(e["part"], e["real"]) for e in entries}


def parse_freeze(text: str | None) -> datetime | None:
    """'YYYY-MM-DD HH:MM' -> datetime (None when no freeze is set)."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M") if text else None


def drop_holdout(groups: dict, held: set[str], parts: list[int]) -> list[str]:
    """Remove the held-out groups from the training groups. A held-out group whose part is being trained on
    but is not found is an error (a typo there would silently leak the demo videos into training)."""
    missing = sorted(k for k in held if int(k.split("/")[0]) in parts and k not in groups)
    if missing:
        raise SystemExit(f"hold-out group(s) not found in the training parts: {missing}; check app_holdout.json")
    return sorted(k for k in held if groups.pop(k, None) is not None)

BASE = dict(arch="vit_b16", linear_probe=False, parts=None, epochs=20, patience=5, smooth=3,
            frames_per_video=4, warmup_epochs=1, min_lr=1e-6, aug="light", unpaired=False,
            include_flagged=False, val_frames=8, val_batch=128, seed=0, workers=4, note="")
KEYS = sorted(set(BASE) | {k for d in ARCH_DEFAULTS.values() for k in d})


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="run name -> runs\\<name>\\ (required except for --bench)")
    ap.add_argument("--config", type=Path, help="YAML file with any of the settings below")
    ap.add_argument("--arch", choices=sorted(ARCH_DEFAULTS))
    ap.add_argument("--linear-probe", action="store_true", default=None, help="CLIP: train the head only (E1)")
    ap.add_argument("--parts", help="training parts: 'dev' (00-05, Stage A) or e.g. 0,1,2 "
                                    "(default: every extracted training part, 00-07 and 10-19)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--patience", type=int, help="early stop after N epochs without smoothed improvement (0 = off)")
    ap.add_argument("--smooth", type=int, help="epochs averaged for early stopping")
    ap.add_argument("--batch", type=int, help="images per step = pairs x frames-per-video x 2")
    ap.add_argument("--frames-per-video", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--warmup-epochs", type=float)
    ap.add_argument("--min-lr", type=float)
    ap.add_argument("--aug", choices=PRESETS)
    ap.add_argument("--loss", choices=["gend", "ce"], help="gend = CE + 0.1 align + 0.5 unif; ce = CE only")
    ap.add_argument("--label-smoothing", type=float)
    ap.add_argument("--unpaired", action="store_true", default=None, help="E4b: fakes from other originals")
    ap.add_argument("--include-flagged", action="store_true", default=None, help="E4c: train on flagged fakes too")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--val-frames", type=int, help="evenly spaced part-08 frames per video each epoch (0 = all)")
    ap.add_argument("--val-batch", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int, help="DataLoader workers (0 if the loader hangs)")
    ap.add_argument("--note")
    ap.add_argument("--resume", action="store_true", help="continue runs\\<name>\\last.pt")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N epochs in this session (resume test)")
    ap.add_argument("--freeze", help="freeze time 'YYYY-MM-DD HH:MM' (PLAN 10.3): an epoch ending at or after it "
                                     "is the last one; best.pt stays the final model")
    ap.add_argument("--bench", type=int, default=0, help="run N training steps, report img/s and peak VRAM, exit")
    ap.add_argument("--max-steps", type=int, default=0, help=argparse.SUPPRESS)  # debug: cap steps per epoch
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV, help=argparse.SUPPRESS)
    ap.add_argument("--holdout", type=Path, default=HOLDOUT_JSON, help=argparse.SUPPRESS)
    ap.add_argument("--min-free-gb", type=float, default=6.0,
                    help="free disk is checked at the start of every epoch; below this the run stops after that "
                         "epoch (resume later)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def resolve_config(args) -> dict:
    """defaults < arch defaults < YAML < CLI."""
    y = {}
    if args.config:
        y = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        unknown = set(y) - set(KEYS)
        if unknown:
            raise SystemExit(f"unknown keys in {args.config}: {sorted(unknown)}")
    cli = {k: v for k, v in vars(args).items() if k in KEYS and v is not None}
    arch = cli.get("arch") or y.get("arch") or BASE["arch"]
    cfg = dict(BASE) | ARCH_DEFAULTS[arch] | y | cli
    cfg["arch"] = arch
    if isinstance(cfg["parts"], str):
        if cfg["parts"].strip().lower() == "dev":
            cfg["parts"] = list(DEV_PARTS)
        else:
            cfg["parts"] = [int(p) for p in cfg["parts"].split(",") if p.strip()]
    if cfg["parts"] is not None:
        cfg["parts"] = sorted(int(p) for p in cfg["parts"])
    per_pair = 2 * cfg["frames_per_video"]
    if cfg["batch"] % per_pair:
        raise SystemExit(f"batch {cfg['batch']} must be a multiple of 2 x frames_per_video = {per_pair}")
    return cfg


def main(argv=None):
    args = parse_args(argv)
    bench = args.bench > 0
    if not bench and not args.name:
        raise SystemExit("--name is required (or use --bench N)")
    name = args.name or f"bench_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.runs_dir / name
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    log, log_path = setup_logger(f"train_{name}")
    t_start = time.time()
    device = args.device

    # ---------------------------------------------------------------- config
    ck = None
    if args.resume:
        if not last_path.exists():
            raise SystemExit(f"--resume: {last_path} not found")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        cfg["workers"] = args.workers if args.workers is not None else cfg["workers"]
        log.info(f"resuming {name} after epoch {ck['epoch'] + 1} (settings come from the checkpoint)")
    else:
        if not bench and last_path.exists():
            raise SystemExit(f"{run_dir} already has a run. Use --resume to continue it or pick a new --name.")
        cfg = resolve_config(args)

    # ---------------------------------------------------------------- data (test-set guard first)
    avail = train_parts(done_parts(args.crops_dir))
    parts = cfg["parts"] if cfg["parts"] is not None else avail
    check_split_parts(parts, [VAL_PART])
    missing = [p for p in parts if p not in avail]
    if missing:
        raise SystemExit(f"training parts not extracted (no DONE.json): {missing}")
    if not parts:
        raise SystemExit("no training parts extracted")
    if VAL_PART not in done_parts(args.crops_dir):
        raise SystemExit("part 08 (validation) is not extracted; run scripts\\process_batch.py first")
    cfg["parts"] = parts

    seed_everything(cfg["seed"])
    groups = load_groups(parts, args.crops_dir, include_flagged=cfg["include_flagged"])
    dropped = drop_holdout(groups, holdout_keys(args.holdout), parts)
    if dropped:
        log.info(f"app-test hold-out: excluded group(s) {dropped} (real + all its fakes) from training")
    val_groups = list(load_groups([VAL_PART], args.crops_dir, include_flagged=False).values())
    overlap = set(groups) & {g.key for g in val_groups}
    if overlap:
        raise SystemExit(f"train/validation overlap: {sorted(overlap)[:3]}")
    train_g = list(groups.values())
    n_fakes = sum(len(g.fakes) for g in train_g)
    val_items = eval_items(val_groups, cfg["val_frames"])
    n_val_videos = len(val_groups) + sum(len(g.fakes) for g in val_groups)
    log.info(f"train parts {parts}: {len(train_g)} real videos, {n_fakes} usable fakes"
             f"{' (flagged included)' if cfg['include_flagged'] else ''} | val part {VAL_PART:02d}: "
             f"{len(val_groups)} real / {n_val_videos - len(val_groups)} fake videos, {len(val_items)} frames "
             f"({cfg['val_frames'] or 'all'} per video)")

    fpv = cfg["frames_per_video"]
    sampler = PairedBatchSampler(train_g, cfg["batch"] // (2 * fpv), fpv, paired=not cfg["unpaired"],
                                 seed=cfg["seed"])
    steps_per_epoch = len(sampler)
    if steps_per_epoch == 0:
        raise SystemExit(f"batch {cfg['batch']} is larger than one epoch of pairs ({len(train_g)} reals)")
    start_epoch = ck["epoch"] + 1 if ck else 0
    gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + start_epoch)
    loader = DataLoader(ItemDataset(build_augment(cfg["aug"])), batch_sampler=sampler, collate_fn=collate,
                        num_workers=cfg["workers"], pin_memory=device == "cuda",
                        persistent_workers=cfg["workers"] > 0, generator=gen)

    # ---------------------------------------------------------------- model and optimiser
    log.info(f"model {cfg['arch']}{' linear probe' if cfg['linear_probe'] else ''} (first use downloads weights)")
    model = build_model(cfg["arch"], linear_probe=cfg["linear_probe"], grad_ckpt=cfg["grad_ckpt"],
                        pretrained=cfg["arch"] != "tiny").to(device)
    total, trainable = model.param_counts()
    qg = check_quickgelu(model)
    log.info(f"parameters: total {total:,}, trainable {trainable:,} ({100 * trainable / total:.3f}%)"
             + ("" if qg is None else f" | QuickGELU {'active' if qg else 'NOT active'}"))
    if qg is False:
        raise SystemExit("QuickGELU is not active: OpenAI CLIP weights would be used with the wrong activation")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = cfg["arch"] == "effb4"
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999))
    total_steps = cfg["epochs"] * steps_per_epoch
    warm = int(round(cfg["warmup_epochs"] * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total_steps, warm, cfg["lr"], cfg["min_lr"]))
    loss_cfg = loss_for(cfg["loss"], cfg["label_smoothing"])
    stopper = EarlyStop(cfg["patience"], cfg["smooth"])
    history = []
    if ck:
        load_trainable(model, ck["trainable"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        stopper.load_state_dict(ck["stopper"])
        history = ck["history"]
        set_rng_state(ck["rng"])
        del ck
    log.info(f"batch {cfg['batch']} ({cfg['batch'] // (2 * fpv)} pairs x {fpv} frames x 2), {steps_per_epoch} "
             f"steps/epoch, lr {cfg['lr']:g} (warm-up {warm} steps, cosine to {cfg['min_lr']:g}), wd "
             f"{cfg['weight_decay']:g}, loss {cfg['loss']}, aug {cfg['aug']}, "
             f"{'unpaired' if cfg['unpaired'] else 'paired'}, seed {cfg['seed']}, workers {cfg['workers']}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------------- bench mode
    if bench:
        bar = tqdm(total=args.bench, desc=f"bench {cfg['arch']} batch {cfg['batch']}", unit="step", dynamic_ncols=True)
        st = dict(steps=0, images=0, seconds=0.0)
        try:
            ep = 0
            while st["steps"] < args.bench:
                sampler.set_epoch(ep)
                r = train_epoch(model, loader, opt, sched, params, loss_cfg, device, bar=bar,
                                max_steps=args.bench - st["steps"])
                for k in st:
                    st[k] += r[k]
                ep += 1
        finally:
            bar.close()
        st["img_s"] = st["images"] / max(1e-9, st["seconds"])
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
        verdict = "ok" if peak <= VRAM_LIMIT_GB else f"above {VRAM_LIMIT_GB} GB: halve the batch"
        append_csv(BENCH_CSV, ["when", "arch", "batch", "grad_ckpt", "steps", "img_s", "peak_vram_gb"],
                   dict(when=datetime.now().isoformat(timespec="seconds"), arch=cfg["arch"], batch=cfg["batch"],
                        grad_ckpt=cfg["grad_ckpt"], steps=st["steps"], img_s=st["img_s"], peak_vram_gb=peak))
        log.info("")
        log.info("==== BENCH SUMMARY ====")
        log.info(f"{cfg['arch']} batch {cfg['batch']} grad_ckpt {cfg['grad_ckpt']}: {st['steps']} steps, "
                 f"{st['img_s']:.1f} img/s, peak VRAM {peak:.2f} GB ({verdict})")
        log.info(f"epoch estimate on {len(train_g)} reals: {steps_per_epoch * cfg['batch'] / st['img_s'] / 60:.1f} min")
        log.info(f"free disk: {free_gb(PROJECT_ROOT):.1f} GB | recorded in {BENCH_CSV}")
        log.info("NEXT: paste this summary back.")
        log.info("=======================")
        return

    # ---------------------------------------------------------------- train
    if not args.resume:
        dump_json(run_dir / "config.json", dict(cfg, name=name, started=datetime.now().isoformat(timespec="seconds"),
                                                train_reals=len(train_g), train_fakes=n_fakes,
                                                holdout_excluded=dropped,
                                                val_videos=n_val_videos, steps_per_epoch=steps_per_epoch))
    log.info(f"run folder: {run_dir} | log: {log_path}")
    stop_reason, epochs_this_session, img_s_all = "finished all epochs", 0, []
    freeze_at = parse_freeze(args.freeze)
    try:
        for epoch in range(start_epoch, cfg["epochs"]):
            sampler.set_epoch(epoch)
            tag = f"epoch {epoch + 1}/{cfg['epochs']}"
            free_now = disk_free_gb()
            low_disk = free_now < args.min_free_gb
            log.info(f"{tag}: free disk {free_now:.1f} GB"
                     + (f" (below {args.min_free_gb:g} GB: stopping after this epoch)" if low_disk else ""))
            bar = tqdm(total=steps_per_epoch, desc=f"{tag} train", unit="step", dynamic_ncols=True)
            tr = train_epoch(model, loader, opt, sched, params, loss_cfg, device, bar=bar, max_steps=args.max_steps)
            bar.close()
            img_s_all.append(tr["img_s"])

            t_val = time.time()
            vbar = tqdm(total=math.ceil(len(val_items) / cfg["val_batch"]), desc=f"{tag} validate part 08",
                        unit="batch", dynamic_ncols=True)
            m, (vids, vy, vs) = validate(model, val_items, device, batch_size=cfg["val_batch"], progress=vbar)
            vbar.close()
            val_s = time.time() - t_val
            new_best, stop = stopper.update(epoch, m["auc"])
            if new_best:
                save_best(best_path, model, epoch, cfg, m)
                write_val_scores(run_dir / "val_best.csv", vids, vy, vs)
            row = dict(epoch=epoch + 1, loss=tr["loss"], ce=tr["ce"], align=tr["align"], unif=tr["unif"],
                       val_auc=m["auc"], val_ap=m["ap"], val_eer=m["eer"], val_logloss=m["logloss"],
                       val_acc=m["acc"], val_auc_smooth=float(sum(stopper.history[-stopper.smooth:]) /
                                                              len(stopper.history[-stopper.smooth:])),
                       lr_end=sched.get_last_lr()[0], img_s=tr["img_s"], train_s=tr["seconds"], val_s=val_s,
                       best="*" if new_best else "", checkpoint=str(best_path if new_best else last_path))
            history.append(row)
            if cfg["arch"] != "effb4":  # small trainable tensors for epoch averaging (PLAN 9.6)
                save_epoch(run_dir / f"ep_{epoch + 1:02d}.pt", model, epoch, cfg, m)
            save_last(last_path, model, opt, sched, epoch, stopper, cfg, history)
            (run_dir / "metrics.csv").unlink(missing_ok=True)
            for r in history:
                append_csv(run_dir / "metrics.csv", METRIC_FIELDS, r)
            log.info(f"{tag}: loss {tr['loss']:.4f} (ce {tr['ce']:.4f}, align {tr['align']:.4f}, unif "
                     f"{tr['unif']:.4f}) | part 08 video AUC {m['auc']:.4f}, AP {m['ap']:.4f}, EER {m['eer']:.4f}"
                     f"{' NEW BEST' if new_best else ''} | {tr['img_s']:.0f} img/s, train {tr['seconds']:.0f} s, "
                     f"val {val_s:.0f} s | {best_path if new_best else last_path}")
            epochs_this_session += 1
            if stop:
                stop_reason = (f"early stop: smoothed AUC ({stopper.smooth} epochs) flat for {stopper.patience} "
                               f"epochs")
                break
            if low_disk and epoch + 1 < cfg["epochs"]:
                stop_reason = (f"stopped: free disk {free_now:.1f} GB was below {args.min_free_gb:g} GB at the "
                               f"start of epoch {epoch + 1}; free space, then resume")
                break
            if args.stop_after and epochs_this_session >= args.stop_after and epoch + 1 < cfg["epochs"]:
                stop_reason = f"stopped after {epochs_this_session} epoch(s) as asked (--stop-after)"
                break
            if freeze_at and datetime.now() >= freeze_at and epoch + 1 < cfg["epochs"]:
                stop_reason = f"stopped at the freeze ({args.freeze}) after epoch {epoch + 1}; best.pt is final"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted (Ctrl+C); last.pt holds the last finished epoch"
    finally:
        del loader  # shut persistent workers down before anything else runs

    # ---------------------------------------------------------------- summary
    done = len(history)
    complete = stop_reason.startswith(("finished", "early stop", "stopped at the freeze"))
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    best = history[stopper.best_epoch] if 0 <= stopper.best_epoch < done else None
    minutes = (time.time() - t_start) / 60
    if complete and best:
        append_csv(args.results, RESULT_FIELDS, dict(
            finished=datetime.now().isoformat(timespec="seconds"), name=name, arch=cfg["arch"],
            linear_probe=cfg["linear_probe"], loss=cfg["loss"], aug=cfg["aug"], paired=not cfg["unpaired"],
            include_flagged=cfg["include_flagged"], train_parts=parts, train_reals=len(train_g),
            val_part=VAL_PART, val_videos=n_val_videos, val_frames=cfg["val_frames"], batch=cfg["batch"],
            lr=cfg["lr"], seed=cfg["seed"], epochs_run=done, best_epoch=stopper.best_epoch + 1,
            val_auc=best["val_auc"], val_ap=best["val_ap"], val_eer=best["val_eer"],
            val_logloss=best["val_logloss"],
            img_s=sum(h["img_s"] for h in history) / done, peak_vram_gb=peak, minutes=minutes, note=cfg["note"]))
    log.info("")
    log.info("==== TRAINING SUMMARY ====")
    log.info(f"run {name}: {cfg['arch']}{' linear probe' if cfg['linear_probe'] else ''}, parts {parts}, "
             f"{len(train_g)} train reals, aug {cfg['aug']}, loss {cfg['loss']}, lr {cfg['lr']:g}, seed {cfg['seed']}")
    log.info(f"epochs done {done}/{cfg['epochs']}: {stop_reason}")
    if best:
        log.info(f"best epoch {stopper.best_epoch + 1}: part 08 video AUC {best['val_auc']:.4f}, AP "
                 f"{best['val_ap']:.4f}, EER {best['val_eer']:.4f}, log loss {best['val_logloss']:.4f} "
                 f"({cfg['val_frames'] or 'all'} frames per video; DFDC numbers are optimistic: shared actors)")
        log.info("AUC by epoch: " + " ".join(f"{h['val_auc']:.3f}" for h in history))
    if img_s_all:
        log.info(f"speed {sum(img_s_all) / len(img_s_all):.1f} img/s | peak VRAM {peak:.2f} GB | time {minutes:.1f} min")
    log.info(f"checkpoints: {best_path} (best), {last_path} (resume)")
    if complete and best:
        log.info(f"results row appended to {args.results}")
    log.info(f"free disk: {free_gb(PROJECT_ROOT):.1f} GB")
    if complete:
        log.info("NEXT: paste this summary back.")
    else:
        log.info(f"NEXT: to continue: python scripts\\train.py --name {name} --resume")
    log.info("==========================")


if __name__ == "__main__":
    main()
