"""Average the trainable tensors of a CLIP run's best epochs into avg.pt and keep it only if it wins (PLAN 9.6).

Picks the --k epochs with the highest part-08 AUC (runs\\<run>\\metrics.csv), averages their ep_NN.pt tensors,
scores avg.pt (and best.pt, if not scored yet) on the part-08 views, and uses avg.pt in place of best.pt only if
its val_robust beats best.pt's by at least --margin (0.003 = 0.3 AUC points, the PLAN 9.5 rule for a more
complex option). Writes runs\\<run>\\avg_decision.json and avg_epochs, avg_val_robust, avg_used in results.csv.
Not for EfficientNet-B4 (no per-epoch files). Start only when no training or extraction job is running.

  python scripts\\average_ckpt.py --run a1_light
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from unmask.common import CROPS_DIR, PROJECT_ROOT, RUNS_DIR, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402
from unmask.train.loop import update_result  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_val import RESULTS_CSV, run_eval  # noqa: E402

MARGIN = 0.003


def top_epochs(run_dir: Path, k: int) -> list[int]:
    """1-based epoch numbers of the k best part-08 AUCs that have an ep_NN.pt file."""
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("val_auc")]
    rows = [r for r in rows if (run_dir / f"ep_{int(r['epoch']):02d}.pt").exists()]
    rows.sort(key=lambda r: float(r["val_auc"]), reverse=True)
    return sorted(int(r["epoch"]) for r in rows[:k])


def average(paths: list[Path]) -> dict:
    """Mean of each floating tensor over the checkpoints; other tensors are taken from the first."""
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    keys = set(cks[0]["trainable"])
    for c in cks[1:]:
        if set(c["trainable"]) != keys:
            raise KeyError("epoch checkpoints hold different tensors")
    avg = {}
    for k in sorted(keys):
        t = cks[0]["trainable"][k]
        avg[k] = (torch.stack([c["trainable"][k].float() for c in cks]).mean(0).to(t.dtype)
                  if t.is_floating_point() else t.clone())
    base = cks[0]
    return dict(kind="avg", config=base["config"], arch=base["arch"], linear_probe=base["linear_probe"],
                trainable=avg, epochs=[c["epoch"] + 1 for c in cks])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=3, help="number of best epochs to average")
    ap.add_argument("--margin", type=float, default=MARGIN, help="val_robust gain avg.pt needs over best.pt")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV, help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    log, log_path = setup_logger(f"average_ckpt_{args.run}")
    t0 = time.time()
    run_dir = args.runs_dir / args.run
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("arch") == "effb4":
        raise SystemExit("epoch averaging is for the CLIP models only (PLAN 9.6)")

    epochs = top_epochs(run_dir, args.k)
    if len(epochs) < 2:
        raise SystemExit(f"need at least 2 epoch checkpoints (ep_NN.pt) in {run_dir}; found {len(epochs)}")
    avg = average([run_dir / f"ep_{e:02d}.pt" for e in epochs])
    torch.save(avg, run_dir / "avg.pt")
    log.info(f"{args.run}: averaged epochs {epochs} -> {run_dir / 'avg.pt'}")

    kw = dict(runs_dir=args.runs_dir, crops_dir=args.crops_dir, results=args.results, device=args.device, log=log)
    best_json = run_dir / "eval_val_best.json"
    best = json.loads(best_json.read_text()) if best_json.exists() else run_eval(args.run, "best", **kw)
    avg_res = run_eval(args.run, "avg", **kw)
    gain = avg_res["val_robust"] - best["val_robust"]
    use = gain >= args.margin
    decision = dict(run=args.run, epochs=epochs, best_val_robust=best["val_robust"],
                    avg_val_robust=avg_res["val_robust"], gain=gain, margin=args.margin, use_avg=use,
                    checkpoint=str(run_dir / ("avg.pt" if use else "best.pt")))
    (run_dir / "avg_decision.json").write_text(json.dumps(decision, indent=1))
    row = update_result(args.results, args.run, dict(avg_epochs=" ".join(map(str, epochs)),
                                                     avg_val_robust=avg_res["val_robust"], avg_used=use))
    log.info("")
    log.info("==== AVERAGE_CKPT SUMMARY ====")
    log.info(f"run {args.run}: epochs {epochs} averaged | val_robust best.pt {best['val_robust']:.4f}, "
             f"avg.pt {avg_res['val_robust']:.4f} (gain {100 * gain:+.2f} points, needs {100 * args.margin:.1f})")
    log.info(f"decision: use {'avg.pt' if use else 'best.pt'} | results.csv row {'updated' if row else 'not found'}")
    log.info(f"time {(time.time() - t0) / 60:.1f} min | free disk {free_gb(PROJECT_ROOT):.1f} GB | log {log_path}")
    log.info("NEXT: paste this summary back.")
    log.info("==============================")
    return decision


if __name__ == "__main__":
    main()
