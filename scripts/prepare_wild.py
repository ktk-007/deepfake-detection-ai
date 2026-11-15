"""WildDeepfake test split -> data\\crops\\wild_test\\ (PLAN 7.3). Test only: never trained or tuned on.

Streams the selected members of each tar into memory (nothing is extracted to disk), keeps 32 evenly spaced
frames per clip (one clip = one sequence folder), resizes to 224 x 224 if needed, saves JPEG 95, writes
index.csv (video, label, n_frames, source_tar, n_source_frames, status), runs gates W1-W5 and writes DONE.json.
Resumable by tar. The tar files are only read and never deleted here.

  python scripts\\prepare_wild.py
  python scripts\\prepare_wild.py --limit-tars 2 --out <scratch folder>   (quick trial, no DONE.json)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from unmask.common import CROPS_DIR, DATA_DIR, PROJECT_ROOT, setup_logger  # noqa: E402
from unmask.data.batch import dir_size, free_gb  # noqa: E402
from unmask.data.wild import (WildConfig, append_rows, config_dict, list_tars, process_tar,  # noqa: E402
                              read_wild_index, run_wild_gates, sample_sheet, tar_key)

WILD_ROOT = DATA_DIR / "test" / "wilddeepfake" / "deepfake_in_the_wild"
EST_MB = 806 * 32 * 13 / 1024  # 806 clips x 32 frames x about 13 KB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=WILD_ROOT)
    ap.add_argument("--out", type=Path, default=CROPS_DIR / "wild_test")
    ap.add_argument("--limit-tars", type=int, default=0, help="trial: first N tars of each class, no DONE.json")
    args = ap.parse_args()
    log, log_path = setup_logger("prepare_wild")
    cfg = WildConfig()
    t0 = time.time()
    out = args.out
    trial = args.limit_tars > 0

    def summary(lines, nxt):
        log.info("")
        log.info("==== WILDDEEPFAKE SUMMARY ====")
        for line in lines:
            log.info(line)
        log.info(f"crops: {dir_size(out) / 2**20:.0f} MB in {out} | free disk: {free_gb(PROJECT_ROOT):.1f} GB | "
                 f"time {(time.time() - t0) / 60:.1f} min")
        log.info("NEXT: " + nxt)
        log.info(f"log: {log_path}")
        log.info("==============================")

    if (out / "DONE.json").exists() and not trial:
        summary(["DONE.json exists, nothing to do"], "look at _samples.jpg, then decide on deleting the tar files.")
        return

    tars = list_tars(args.src)
    if trial:
        tars = [t for lab in ("FAKE", "REAL") for t in [x for x in tars if x[1] == lab][:args.limit_tars]]
    n_fake = sum(1 for _, lab in tars if lab == "FAKE")
    log.info(f"source: {args.src} | tars: {len(tars)} ({n_fake} fake, {len(tars) - n_fake} real)")
    log.info(f"settings: frames={cfg.frames} size={cfg.size} jpeg_q={cfg.quality} short_below={cfg.short_below}")
    if not tars:
        summary(["no tar files found"], f"check that the tars are in {args.src}")
        return
    have = free_gb(PROJECT_ROOT)
    if have < 3 + EST_MB / 1024:
        summary([f"STOP: free disk {have:.1f} GB < 3 GB + about {EST_MB:.0f} MB of crops"], "free disk space first.")
        return
    log.info(f"free disk {have:.1f} GB, about {EST_MB:.0f} MB of crops expected")

    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.csv"
    rows = read_wild_index(index)
    done_tars = {r["source_tar"] for r in rows}
    if done_tars:
        log.info(f"resuming: {len(done_tars)} tars already in index.csv")

    stats_file = out / "_read_stats.json"  # selected/read counts survive a resume
    tot = json.loads(stats_file.read_text()) if stats_file.exists() and done_tars else dict(
        selected=0, read_ok=0, png=0, skipped=0)
    failed = []
    bar = tqdm(tars, desc="tars", unit="tar", dynamic_ncols=True)
    for path, label in bar:
        if tar_key(path) in done_tars:
            continue
        bar.set_postfix_str(tar_key(path))
        try:
            new_rows, st = process_tar(path, label, out, cfg)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{tar_key(path)}: {e}")
            log.info(f"FAILED to read {tar_key(path)}: {e}")
            continue
        append_rows(index, new_rows)
        rows += new_rows
        for k in tot:
            tot[k] += st[k]
        stats_file.write_text(json.dumps(tot))
        bar.set_postfix(clips=len(rows), frames=tot["read_ok"])
    bar.close()

    log.info("stage: sample sheet and gates")
    sample_sheet(out, rows, seed=0, size=cfg.size)
    gates = run_wild_gates(out, rows, len(tars), failed, tot["selected"], tot["read_ok"], size=cfg.size)
    for g in gates:
        log.info(f"{g['gate']} {'PASS' if g['ok'] else 'FAIL'}: {g['detail']}")
    passed = all(g["ok"] for g in gates)

    lines = []
    per = {}
    for lab in ("REAL", "FAKE"):
        rs = [r for r in rows if r["label"] == lab]
        nf = sorted(int(r["n_source_frames"]) for r in rs)
        per[lab.lower()] = dict(
            tars=len({r["source_tar"] for r in rs}), clips=len(rs),
            ok=sum(r["status"] == "ok" for r in rs), short=sum(r["status"] == "short" for r in rs),
            error=sum(r["status"] == "error" for r in rs), frames_saved=sum(int(r["n_frames"]) for r in rs),
            source_frames_min=nf[0] if nf else 0, source_frames_median=nf[len(nf) // 2] if nf else 0,
            source_frames_max=nf[-1] if nf else 0)
        p = per[lab.lower()]
        lines.append(f"{lab.lower()}: {p['clips']} clips from {p['tars']} tars (ok {p['ok']}, short {p['short']}, "
                     f"error {p['error']}), {p['frames_saved']} frames saved; source frames per clip "
                     f"min {p['source_frames_min']} / median {p['source_frames_median']} / max {p['source_frames_max']}")
    lines.append(f"members listed {tot['png']} PNG ({tot['skipped']} skipped), selected {tot['selected']}, "
                 f"read ok {tot['read_ok']} | crop size {cfg.size}x{cfg.size} JPEG {cfg.quality}")
    lines.append(f"gates: {'PASS' if passed else 'FAIL'}")

    if trial:
        summary(lines + ["trial run (--limit-tars): DONE.json not written"],
                "trial only. Run again without --limit-tars and --out.")
        return
    if not passed:
        summary(lines + ["STOP: " + "; ".join(f"{g['gate']} ({g['detail']})" for g in gates if not g["ok"])],
                "paste this summary back. Tar files kept.")
        return
    src_bytes = sum(p.stat().st_size for p, _ in tars)
    info = dict(dataset="wilddeepfake_test", gates_pass=True, gates=gates, classes=per,
                members=tot, settings=config_dict(cfg), clip_definition="one sequence folder per clip",
                bootstrap_unit="source_tar", source=str(args.src), source_bytes=src_bytes,
                crops_bytes=dir_size(out), seconds=round(time.time() - t0, 1),
                finished=datetime.now().isoformat(timespec="seconds"))
    (out / "DONE.json").write_text(json.dumps(info, indent=1))
    summary(lines + ["DONE.json written. Tar files kept (ask before deleting, about "
                     f"{src_bytes / 1e9:.1f} GB)."],
            "open data\\crops\\wild_test\\_samples.jpg, paste this summary back; then decide on deleting the tars.")


if __name__ == "__main__":
    main()
