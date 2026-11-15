"""Process every DFDC part zip in "zip data\\": pre-flight, extract, gates, DONE.json, zip deletion (PLAN 7.2).

Rerun the same command for every batch. Parts with DONE.json are skipped. Extraction resumes by group.
Zips are deleted only for parts that passed all gates, and only after data\\pipeline_ok.json exists
(written by scripts\\smoke_train.py).

  python scripts\\process_batch.py
  python scripts\\process_batch.py --limit-groups 3     (quick trial; does not write DONE.json)
"""
import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unmask.common import CROPS_DIR, PIPELINE_OK, PROJECT_ROOT, ZIP_DIR, setup_logger  # noqa: E402
from unmask.data.batch import (delete_done_zips, dir_size, done_info, done_parts, free_gb,  # noqa: E402
                               next_message, part_dir)
from unmask.data.dfdc_zip import build_groups, find_part_zips, load_metadata  # noqa: E402
from unmask.data.extract import (config_for_part, estimate_mb, extract_part, planned_videos,  # noqa: E402
                                 score_percentiles)
from unmask.data.gates import gates_pass, make_sample_sheet, run_gates  # noqa: E402
from unmask.data.index import read_index  # noqa: E402


def preflight(part: int, path: Path, log) -> tuple[bool, str, float]:
    try:
        with zipfile.ZipFile(path) as zf:
            meta, prefix, _ = load_metadata(zf)
            _, fakes_of, _ = build_groups(meta, set(zf.namelist()), prefix)
    except Exception as e:  # noqa: BLE001
        return False, f"zip or metadata.json unreadable: {e}", 0.0
    if not fakes_of:
        return False, "no real videos found in metadata.json", 0.0
    cfg = config_for_part(part)
    est = estimate_mb(planned_videos(fakes_of, cfg.max_fakes), cfg.frames)
    need = 3 + est / 1024
    have = free_gb(PROJECT_ROOT)
    if have < need:
        return False, f"free disk {have:.1f} GB < needed {need:.1f} GB (3 GB + about {est:.0f} MB crops)", est
    return True, f"ok, about {est:.0f} MB of crops expected, free disk {have:.1f} GB", est


def part_line(part: int, rows: list[dict], gates_ok: bool | None, out_dir: Path) -> str:
    reals = [r for r in rows if r["label"] == "REAL"]
    n_ok = sum(1 for r in reals if r["status"] == "ok")
    scored = [r for r in rows if r["label"] == "FAKE" and r["status"] == "ok"]
    n_flag = sum(1 for r in scored if str(r["flagged"]) == "1")
    g = "PASS" if gates_ok else ("FAIL" if gates_ok is False else "not run")
    return (f"part {part:02d}: reals ok {n_ok}/{len(reals)}, usable fakes {len(scored) - n_flag}, "
            f"flagged {100 * n_flag / max(1, len(scored)):.1f}% ({n_flag}), "
            f"crops {dir_size(out_dir) / 2**20:.0f} MB, gates {g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-groups", type=int, default=0, help="trial run: first N groups only, no DONE.json")
    args = ap.parse_args()
    log, log_path = setup_logger("process_batch")
    t0 = time.time()

    zips = find_part_zips(ZIP_DIR)
    done = done_parts()
    log.info(f'zips found in "{ZIP_DIR}": ' + (", ".join(p.name for p in zips.values()) or "none"))
    log.info(f"parts already done: {sorted(done) or 'none'} | free disk {free_gb(PROJECT_ROOT):.1f} GB")

    detector = None
    lines, failures = [], []
    for part, path in zips.items():
        out_dir = part_dir(part)
        if part in done:
            info = done_info(part)
            log.info(f"part {part:02d}: DONE.json exists, skipping extraction")
            lines.append(part_line(part, read_index(out_dir / "index.csv"), info.get("gates_pass"), out_dir))
            continue

        ok, msg, est = preflight(part, path, log)
        log.info(f"part {part:02d}: pre-flight {msg}")
        if not ok:
            failures.append(f"part {part:02d}: pre-flight failed: {msg}")
            lines.append(f"part {part:02d}: pre-flight FAIL ({msg}), zip kept")
            continue

        if detector is None:
            import torch

            if not torch.cuda.is_available():
                log.info("CUDA is not available. Stopping: face detection needs the GPU.")
                return
            from unmask.data.faces import FaceDetector

            detector = FaceDetector(device="cuda")

        cfg = config_for_part(part)
        log.info(f"part {part:02d}: extracting, frames={cfg.frames} max_fakes={cfg.max_fakes} "
                 f"max_tries={cfg.max_tries} margin={cfg.margin} size={cfg.size} jpeg_q={cfg.quality} "
                 f"threshold={cfg.threshold} seed={cfg.seed}")
        stats = extract_part(path, part, out_dir, cfg, detector, log=log, limit_groups=args.limit_groups)

        rows = read_index(out_dir / "index.csv")
        make_sample_sheet(out_dir, rows, seed=cfg.seed)
        results = run_gates(out_dir, rows, size=cfg.size)
        passed = gates_pass(results)
        for g in results:
            flag = "PASS" if g["ok"] else "FAIL"
            if g["ok"] and g["warn"]:
                flag = "PASS (warning)"
            log.info(f"part {part:02d}: {g['gate']} {flag}: {g['detail']}")
        pct = score_percentiles(rows)
        if pct:
            log.info(f"part {part:02d}: fake score min/10/25/50/75/90th pct: " + " / ".join(map(str, pct)))
        log.info(f"part {part:02d}: {stats['videos_read']} videos read in {stats['seconds'] / 60:.1f} min")

        if args.limit_groups:
            log.info(f"part {part:02d}: trial run (--limit-groups), DONE.json not written")
            passed_final = None
        elif passed:
            reals = [r for r in rows if r["label"] == "REAL"]
            scored = [r for r in rows if r["label"] == "FAKE" and r["status"] == "ok"]
            n_flag = sum(1 for r in scored if str(r["flagged"]) == "1")
            per_real = Counter(r["original"] for r in scored)  # evaluation groups by original video (PLAN 7.1)
            info = dict(
                part=part, gates_pass=True, gates=results,
                reals=len(reals), reals_ok=sum(1 for r in reals if r["status"] == "ok"),
                fakes_scored=len(scored), fakes_usable=len(scored) - n_flag, fakes_flagged=n_flag,
                max_fakes_per_real=max(per_real.values(), default=0),
                score_percentiles=pct, settings=stats["config"],
                zip_name=path.name, zip_bytes=path.stat().st_size,
                crops_bytes=dir_size(out_dir), extraction_seconds=stats["seconds"],
                finished=datetime.now().isoformat(timespec="seconds"),
            )
            (out_dir / "DONE.json").write_text(json.dumps(info, indent=1))
            passed_final = True
        else:
            failed = [f"{g['gate']} ({g['detail']})" for g in results if not g["ok"]]
            failures.append(f"part {part:02d}: gates failed: " + "; ".join(failed))
            passed_final = False
        lines.append(part_line(part, rows, passed_final, out_dir))

    deleted, zip_status = delete_done_zips(log)
    if not zips:
        log.info('no zips in "zip data\\"')

    done = done_parts()
    crops_gb = dir_size(CROPS_DIR) / 2**30
    log.info("")
    log.info("==== BATCH SUMMARY ====")
    for line in lines:
        log.info(line)
    log.info(f"zips deleted: {zip_status}")
    log.info(f"crops total: {crops_gb:.2f} GB | free disk: {free_gb(PROJECT_ROOT):.1f} GB | "
             f"time {(time.time() - t0) / 60:.1f} min")
    if failures:
        for f in failures:
            log.info(f"STOP: {f}")
        log.info("NEXT: paste this summary back; the failing part's zip is kept.")
    elif args.limit_groups:
        log.info("NEXT: trial run only. Run again without --limit-groups.")
    elif not PIPELINE_OK.exists():
        log.info("NEXT: run the smoke test: python scripts\\smoke_train.py (it deletes the zips on success).")
    else:
        log.info("NEXT: " + next_message(done))
    log.info(f"log: {log_path}")
    log.info("=======================")


if __name__ == "__main__":
    main()
