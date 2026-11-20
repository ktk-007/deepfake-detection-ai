"""Score a finished run on the clean, tight and degraded views of part 08 (PLAN 9.5).

Writes runs\\<run>\\eval_val_<ckpt>.json and, for best.pt, fills val_clean, val_tight, val_degraded and
val_robust (their mean) in the run's row of results\\results.csv. Part 08 only: no test set is touched.
Threaded decoding, no DataLoader workers. Start it only when no training or extraction job is running.

  python scripts\\eval_val.py --run a1_light
  python scripts\\eval_val.py --run a1_light --ckpt avg          (runs\\a1_light\\avg.pt)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from unmask.common import CROPS_DIR, PROJECT_ROOT, RUNS_DIR, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402
from unmask.eval.val_eval import EVAL_FRAMES, evaluate_checkpoint  # noqa: E402
from unmask.eval.views import VIEWS  # noqa: E402
from unmask.train.loop import update_result  # noqa: E402

RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name under runs\\")
    ap.add_argument("--ckpt", default="best", help="best, avg, or a file name inside the run folder")
    ap.add_argument("--frames", type=int, default=EVAL_FRAMES, help="evenly spaced frames per video")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV, help=argparse.SUPPRESS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def ckpt_path(run_dir: Path, ckpt: str) -> tuple[Path, str]:
    if ckpt in ("best", "avg", "last"):
        return run_dir / f"{ckpt}.pt", ckpt
    p = run_dir / ckpt
    return p, p.stem


def run_eval(run: str, ckpt: str = "best", frames: int = EVAL_FRAMES, batch: int = 128, runs_dir: Path = RUNS_DIR,
             crops_dir: Path = CROPS_DIR, results: Path = RESULTS_CSV, device: str = "cuda", log=None) -> dict:
    run_dir = runs_dir / run
    path, tag = ckpt_path(run_dir, ckpt)
    if not path.exists():
        raise SystemExit(f"{path} not found")

    def bar(view, n):
        return tqdm(total=n, desc=f"{run} {tag}: part 08 {view} view", unit="batch", dynamic_ncols=True)

    res = evaluate_checkpoint(path, device, crops_dir, frames, batch, progress=bar)
    res["run"], res["ckpt"] = run, tag
    res["finished"] = datetime.now().isoformat(timespec="seconds")
    (run_dir / f"eval_val_{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    row_done = False
    if tag == "best":
        row_done = update_result(results, run, {f"val_{v}": res[v]["auc"] for v in VIEWS}
                                 | {"val_robust": res["val_robust"]})
    if log:
        for v in VIEWS:
            m = res[v]
            log.info(f"{run} {tag} {v:9s}: video AUC {m['auc']:.4f}, AP {m['ap']:.4f}, EER {m['eer']:.4f} "
                     f"({m['n_real']} real / {m['n_fake']} fake videos, {m['seconds']:.0f} s)")
        log.info(f"{run} {tag} val_robust (mean of the three AUCs): {res['val_robust']:.4f}")
        if tag == "best":
            log.info(f"results.csv row {'updated' if row_done else 'NOT found (run incomplete?)'}: {results}")
    res["row_updated"] = row_done
    return res


def main(argv=None):
    args = parse_args(argv)
    log, log_path = setup_logger(f"eval_val_{args.run}")
    t0 = time.time()
    log.info(f"scoring runs\\{args.run}\\{args.ckpt} on part 08 views {', '.join(VIEWS)}, {args.frames} frames per "
             f"video (validation only; no test set is loaded)")
    res = run_eval(args.run, args.ckpt, args.frames, args.batch, args.runs_dir, args.crops_dir, args.results,
                   args.device, log)
    log.info("")
    log.info("==== EVAL_VAL SUMMARY ====")
    ep = res.get("epoch")                       # stored 0-based in the checkpoint; printed 1-based like train.py
    log.info(f"run {args.run}, checkpoint {res['ckpt']} (epoch {ep + 1 if isinstance(ep, int) else ep}): "
             + ", ".join(f"{v} {res[v]['auc']:.4f}" for v in VIEWS) + f" | val_robust {res['val_robust']:.4f}")
    log.info(f"time {(time.time() - t0) / 60:.1f} min | free disk {free_gb(PROJECT_ROOT):.1f} GB | log {log_path}")
    log.info("NEXT: paste this summary back.")
    log.info("==========================")


if __name__ == "__main__":
    main()
