"""Run the night queue (PLAN 10.2, 10.3): one training run at a time, each followed by its part-08 checks.

  python scripts\\run_queue.py configs\\queue_finals.yaml
  python scripts\\run_queue.py configs\\queue_finals.yaml --dry-run      (print the plan, run nothing)

The queue file names the freeze time and the runs in order. Each run is scripts\\train.py with its config plus
the overrides in `set` (passed as command-line flags), and --freeze, so the run in progress at the freeze stops
after its current epoch and keeps its best.pt. After each run: eval_val.py, then average_ckpt.py (CLIP only).
No run starts at or after the freeze, or with less than --min-start-gb free; those are listed as planned, not
run. A run that already finished (best.pt and a results row) is skipped; one with last.pt but not finished is
resumed. One process at a time: never start this while another training, extraction or evaluation job runs.
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from unmask.common import PROJECT_ROOT, RUNS_DIR, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"


def parse_freeze(text: str) -> datetime:
    return datetime.strptime(str(text), "%Y-%m-%d %H:%M")


def load_queue(path: Path) -> tuple[datetime, list[dict]]:
    q = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    freeze = parse_freeze(q["freeze"])
    runs = q["runs"]
    names = [r["name"] for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate run names in {path}")
    for r in runs:
        if not (PROJECT_ROOT / r["config"]).exists() and not Path(r["config"]).exists():
            raise SystemExit(f"{r['name']}: config {r['config']} not found")
    return freeze, runs


def flags(overrides: dict | None) -> list[str]:
    out = []
    for k, v in (overrides or {}).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out += [flag] if v else [f"--no-{k.replace('_', '-')}"]
        else:
            out += [flag, str(v)]
    return out


def finished(name: str, runs_dir: Path, results: Path) -> bool:
    if not (runs_dir / name / "best.pt").exists() or not results.exists():
        return False
    with open(results, encoding="utf-8", newline="") as f:
        return any(r.get("name") == name for r in csv.DictReader(f))


def result_row(name: str, results: Path) -> dict:
    if not results.exists():
        return {}
    with open(results, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("name") == name]
    return rows[-1] if rows else {}


def commands(run: dict, freeze_text: str, runs_dir: Path) -> list[list[str]]:
    py = sys.executable
    train = [py, str(SCRIPTS / "train.py"), "--name", run["name"]]
    if (runs_dir / run["name"] / "last.pt").exists():
        train += ["--resume"]
    else:
        train += ["--config", str(run["config"])] + flags(run.get("set"))
    train += ["--freeze", freeze_text]
    cmds = [train, [py, str(SCRIPTS / "eval_val.py"), "--run", run["name"]]]
    cfg_path = Path(run["config"]) if Path(run["config"]).is_absolute() else PROJECT_ROOT / run["config"]
    arch = (run.get("set") or {}).get("arch") or yaml.safe_load(cfg_path.read_text(encoding="utf-8")).get("arch")
    if arch != "effb4":
        cmds.append([py, str(SCRIPTS / "average_ckpt.py"), "--run", run["name"]])
    return cmds


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", type=Path, help="queue YAML (freeze + runs)")
    ap.add_argument("--min-start-gb", type=float, default=20.0, help="free disk needed to start a run (PLAN 7.1)")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    ap.add_argument("--disk-wait-checks", type=int, default=10, help="30 s re-checks of free disk before skipping")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None, now=datetime.now, disk=lambda: free_gb(PROJECT_ROOT), run_cmd=None, sleep=time.sleep):
    args = parse_args(argv)
    freeze, runs = load_queue(args.queue)
    freeze_text = freeze.strftime("%Y-%m-%d %H:%M")
    log, log_path = setup_logger("run_queue")
    run_cmd = run_cmd or (lambda c: subprocess.run(c, cwd=PROJECT_ROOT).returncode)
    log.info(f"queue {args.queue}: {len(runs)} runs, freeze {freeze_text} (no run starts at or after it; the run in "
             f"progress stops after its current epoch)")
    report, t0 = [], time.time()
    for i, run in enumerate(runs):
        name = run["name"]
        if finished(name, args.runs_dir, args.results):
            report.append((name, "already finished, skipped"))
            log.info(f"[{i + 1}/{len(runs)}] {name}: already finished, skipped")
            continue
        if now() >= freeze:
            report.append((name, "planned, not run (freeze reached)"))
            continue
        free = disk()
        # Windows releases the previous run's page file over a few minutes: wait up to 5 min before giving up.
        for _ in range(args.disk_wait_checks):
            if free >= args.min_start_gb or now() >= freeze:
                break
            log.info(f"[{i + 1}/{len(runs)}] {name}: free disk {free:.1f} GB below {args.min_start_gb:g} GB; "
                     f"waiting 30 s for the page file to shrink")
            sleep(30)
            free = disk()
        if free < args.min_start_gb:
            report.append((name, f"planned, not run (free disk {free:.1f} GB < {args.min_start_gb:g} GB)"))
            log.info(f"[{i + 1}/{len(runs)}] {name}: free disk {free:.1f} GB below {args.min_start_gb:g} GB; "
                     f"not started")
            continue
        cmds = commands(run, freeze_text, args.runs_dir)
        log.info(f"[{i + 1}/{len(runs)}] {name}: start {now():%H:%M}, free disk {free:.1f} GB")
        for c in cmds:
            log.info("  " + " ".join(f'"{x}"' if " " in x else x for x in c))
        if args.dry_run:
            report.append((name, "dry run"))
            continue
        t_run = time.time()
        try:
            code = run_cmd(cmds[0])
        except Exception as e:                       # the queue never stops on one job's error
            code = f"error {type(e).__name__}: {e}"
        if code != 0 or not (args.runs_dir / name / "best.pt").exists():
            report.append((name, f"FAILED: training exit {code}; see logs/train_{name}_*.log; later runs continue"))
            log.info(f"[{i + 1}/{len(runs)}] {name}: FAILED, training exit {code}; skipping its checks, "
                     f"continuing with the next run")
            continue
        check_errors = []
        for c in cmds[1:]:
            try:
                code = run_cmd(c)
            except Exception as e:
                code = f"error {type(e).__name__}: {e}"
            if code != 0:
                check_errors.append(f"{Path(c[1]).stem} exit {code}")
                log.info(f"[{i + 1}/{len(runs)}] {name}: {Path(c[1]).name} FAILED (exit {code}); continuing")
        row = result_row(name, args.results)
        status = (f"done in {(time.time() - t_run) / 60:.0f} min: epochs {row.get('epochs_run', '?')}, best epoch "
                  f"{row.get('best_epoch', '?')}, part-08 AUC {row.get('val_auc', '?')}, val_robust "
                  f"{row.get('val_robust', '?')}, averaged used {row.get('avg_used', '-') or '-'}") if row else \
            "stopped before its first finished epoch (no results row)"
        if check_errors:
            status += " | check FAILED: " + ", ".join(check_errors)
        report.append((name, status))
        log.info(f"[{i + 1}/{len(runs)}] {name}: {status}")

    log.info("")
    log.info("==== QUEUE SUMMARY ====")
    log.info(f"freeze {freeze_text} | ended {now():%Y-%m-%d %H:%M} | {(time.time() - t0) / 3600:.1f} h")
    for name, status in report:
        log.info(f"{name}: {status}")
    log.info(f"free disk {free_gb(PROJECT_ROOT):.1f} GB | log {log_path}")
    log.info("NEXT: paste this summary back.")
    log.info("=======================")
    return report


if __name__ == "__main__":
    main()
