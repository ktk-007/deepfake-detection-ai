"""Phase 4, step 1 (PLAN 11): everything that is decided on part 08, before any test set is opened.

For every final model: score part 08 clean at all 32 frames, and the tight and degraded views at 16 frames
(with and without a horizontal flip for the candidates of flip averaging, E9). Then, on part 08 only:
  * flip averaging per model: adopted if val_robust rises by >= 0.3 points;
  * temperature T per model (video log loss) and the two verdict thresholds (wrong <= 5% outside the band);
  * the ensemble (E10): calibrated members added greedily while val_robust rises by >= 0.3 points.
Writes results\\phase4\\choices.json (the lock that phase4_test.py requires), results\\calibration.json and
results\\ensemble.json (read by the app). Resumable: scores already saved are reused.
Run only when no training or other GPU job is running.

  python scripts\\phase4_val.py
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unmask.common import CROPS_DIR, PROJECT_ROOT, RUNS_DIR, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402
from unmask.data.dfdc_zip import evenly_spaced  # noqa: E402
from unmask.eval.phase4 import (Scores, choose_band, fit_temperature, load_set, score_videos, subsample,  # noqa: E402
                                val_robust, view_transform)

RESULTS = PROJECT_ROOT / "results"
OUT = RESULTS / "phase4"
# (run, app key, label, flip candidate, ensemble candidate)
FINAL = [
    ("e7_l14", "vit_l14", "E7 CLIP ViT-L/14", True, True),
    ("e2_b16", "vit_b16", "E2 CLIP ViT-B/16", True, True),
    ("e3_effb4", "effb4", "E3 EfficientNet-B4", True, True),
    ("e1_linear_probe", "e1", "E1 linear probe (dev set)", False, False),
    ("e8_gend", "e8", "E8 GenD published (FF++)", False, False),
]
MARGIN = 0.003


def ckpt_for(run_dir: Path) -> Path | None:
    dec = run_dir / "avg_decision.json"
    if dec.exists() and json.loads(dec.read_text(encoding="utf-8")).get("use_avg") and (run_dir / "avg.pt").exists():
        return run_dir / "avg.pt"
    return run_dir / "best.pt" if (run_dir / "best.pt").exists() else None


def subset(s: Scores, n: int) -> Scores:
    """Keep n evenly spaced frames per video (clean view at 16 frames, from the 32-frame scores)."""
    zs, offs = [], [0]
    for i in range(len(s.keys)):
        z = s.z[s.offsets[i]:s.offsets[i + 1]]
        z = z[evenly_spaced(len(z), n)]
        zs.append(z)
        offs.append(offs[-1] + len(z))
    return Scores(s.keys, s.labels, s.groups, np.array(offs), np.concatenate(zs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args(argv)
    out = args.results / "phase4"
    log, log_path = setup_logger("phase4_val")
    t0 = time.time()
    if (out / "choices.json").exists():
        log.info(f"{out / 'choices.json'} exists: part-08 choices are locked. Delete it only to redo step 1.")
        return json.loads((out / "choices.json").read_text(encoding="utf-8"))

    from unmask.train.loop import load_model_checkpoint
    val32 = load_set("val08", args.crops_dir)
    val16 = subsample(val32, 16)
    log.info(f"part 08: {len(val32)} videos ({sum(v.label for v in val32)} fake), validation only; no test set is loaded")

    models = {}
    for run, key, label, flip_cand, ens_cand in FINAL:
        ck = ckpt_for(args.runs_dir / run)
        if ck is None:
            log.info(f"{run}: no checkpoint, skipped (listed as not run)")
            continue
        models[run] = dict(key=key, label=label, flip_cand=flip_cand, ens_cand=ens_cand, ckpt=str(ck))

    scores = {}
    for run, m in models.items():
        model = None
        for flip in ([False, True] if m["flip_cand"] else [False]):
            tag = "_flip" if flip else ""
            for view, vids in (("clean32", val32), ("tight", val16), ("degraded", val16)):
                path = out / f"val_{run}_{view}{tag}.npz"
                if not path.exists():
                    if model is None:
                        model = load_model_checkpoint(Path(m["ckpt"]), args.device)[0]
                        log.info(f"{run}: loaded {m['ckpt']}")
                    n = sum(len(v.frames) for v in vids)
                    bar = tqdm(total=-(-n // args.batch), desc=f"{run} {view}{tag}", unit="batch", dynamic_ncols=True)
                    s = score_videos(model, vids, args.device, transform=view_transform(view.replace("32", "")),
                                     flip=flip, progress=bar, batch=args.batch)
                    bar.close()
                    s.save(path, run=run, view=view, flip=flip)
                scores[(run, view, flip)] = Scores.load(path)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- choices, part 08 only
    def views(run, flip, temperature=1.0):
        c32 = scores[(run, "clean32", False)]
        if flip:
            c32 = Scores(c32.keys, c32.labels, c32.groups, c32.offsets,
                         (c32.z + scores[(run, "clean32", True)].z) / 2)
        out_v = {"clean32": c32, "clean": subset(c32, 16)}
        for v in ("tight", "degraded"):
            s = scores[(run, v, False)]
            if flip:
                s = Scores(s.keys, s.labels, s.groups, s.offsets, (s.z + scores[(run, v, True)].z) / 2)
            out_v[v] = s
        return {k: s.video_probs(temperature) for k, s in out_v.items()}, out_v

    y = scores[(next(iter(models)), "clean32", False)].labels
    choice = {}
    for run, m in models.items():
        p0, _ = views(run, False)
        vr0 = val_robust(p0, y)
        flip, vr = False, vr0
        if m["flip_cand"]:
            p1, _ = views(run, True)
            vr1 = val_robust(p1, y)
            if vr1 - vr0 >= MARGIN:
                flip, vr = True, vr1
            log.info(f"{run}: val_robust {vr0:.4f} without flip, {vr1:.4f} with flip -> flip {'adopted' if flip else 'not adopted'}")
        _, sv = views(run, flip)
        t = fit_temperature(sv["clean32"])
        band = choose_band(sv["clean32"].video_probs(t), y)
        _, sv0 = views(run, False)                # the app scores without flip: its own calibration
        t_app = fit_temperature(sv0["clean32"])
        band_app = choose_band(sv0["clean32"].video_probs(t_app), y)
        pc, _ = views(run, flip, t)
        from sklearn.metrics import roc_auc_score
        choice[run] = dict(key=m["key"], label=m["label"], ckpt=m["ckpt"], flip=flip, val_robust=vr,
                           val_robust_noflip=vr0, val_clean32_auc=float(roc_auc_score(y, pc["clean32"])),
                           view_auc={v: float(roc_auc_score(y, pc[v])) for v in ("clean", "tight", "degraded")},
                           temperature=t, band=band, app=dict(temperature=t_app, **band_app))
        log.info(f"{run}: T {t:.2f}, band {band['low']:.3f}-{band['high']:.3f}{'' if band['fitted'] else ' (fallback)'}, "
                 f"val_robust {vr:.4f}")

    # ensemble (E10): greedy, equal-weight mean of calibrated members
    cands = [r for r, m in models.items() if m["ens_cand"]]
    calib = {r: views(r, choice[r]["flip"], choice[r]["temperature"])[0] for r in cands}
    ens = dict(adopted=False, members=[], val_robust=None, tried=[])
    if cands:
        best = max(cands, key=lambda r: choice[r]["val_robust"])
        members, cur = [best], choice[best]["val_robust"]
        improved = True
        while improved:
            improved, pick = False, None
            for r in cands:
                if r in members:
                    continue
                trial = members + [r]
                vr = val_robust({v: np.mean([calib[x][v] for x in trial], axis=0) for v in ("clean", "tight", "degraded")}, y)
                ens["tried"].append(dict(members=trial, val_robust=vr))
                if vr - cur >= MARGIN and (pick is None or vr > pick[1]):
                    pick = (r, vr)
            if pick:
                members.append(pick[0])
                cur, improved = pick[1], True
        if len(members) > 1:
            pe = np.mean([calib[x]["clean32"] for x in members], axis=0)
            ens.update(adopted=True, members=members, val_robust=cur, band=choose_band(pe, y),
                       best_single=best, best_single_val_robust=choice[best]["val_robust"])
        log.info(f"ensemble: {'adopted ' + ' + '.join(members) + f' (val_robust {cur:.4f})' if ens['adopted'] else 'not adopted (no member added 0.3 points)'}")

    lock = dict(locked_at=datetime.now().isoformat(timespec="seconds"), margin=MARGIN, models=choice, ensemble=ens,
                note="All choices made on DFDC part 08 only, before any test set was opened.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "choices.json").write_text(json.dumps(lock, indent=1), encoding="utf-8")
    cal = {c["key"]: dict(c["app"]) for c in choice.values()}
    if ens["adopted"]:
        cal["ensemble"] = dict(temperature=1.0, low=ens["band"]["low"], high=ens["band"]["high"])
    (args.results / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    (args.results / "ensemble.json").write_text(json.dumps(dict(
        adopted=ens["adopted"], members=[choice[r]["key"] for r in ens["members"]] if ens["adopted"] else []), indent=1),
        encoding="utf-8")

    log.info("")
    log.info("==== PHASE 4 VALIDATION SUMMARY ====")
    for run, c in choice.items():
        log.info(f"{run}: val_robust {c['val_robust']:.4f} (clean {c['view_auc']['clean']:.4f}, tight "
                 f"{c['view_auc']['tight']:.4f}, degraded {c['view_auc']['degraded']:.4f}) | flip {c['flip']} | "
                 f"T {c['temperature']:.2f} | band {c['band']['low']:.3f}-{c['band']['high']:.3f}")
    log.info(f"ensemble: {'+'.join(ens['members']) + ' val_robust ' + format(ens['val_robust'], '.4f') if ens['adopted'] else 'not adopted'}")
    log.info(f"locked: {out / 'choices.json'} | time {(time.time() - t0) / 60:.1f} min | free disk {free_gb(PROJECT_ROOT):.1f} GB | log {log_path}")
    log.info("NEXT: python scripts\\phase4_test.py  (each test set once)")
    log.info("====================================")
    return lock


if __name__ == "__main__":
    main()
