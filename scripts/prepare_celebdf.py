"""Celeb-DF v2 test list -> data\\crops\\celebdf_v2_test\\ (PLAN 7.5). Test only: never trained or tuned on.

Streams only the videos in List_of_testing_videos.txt from the zip (one at a time, temp file deleted), detects
the face per video (MTCNN, GPU), keeps 32 evenly spaced 224 px crops, runs gates C1-C5, writes _samples.jpg and
DONE.json. Resumable. The zip is never deleted here: ask the owner first (PLAN section 4).
Run only when no training, extraction or evaluation job is running.

  python scripts\\prepare_celebdf.py                          (finds the Celeb-DF zip in "zip data\\" or Downloads)
  python scripts\\prepare_celebdf.py --zip "<path to zip>"
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unmask.common import CROPS_DIR, PROJECT_ROOT, setup_logger  # noqa: E402
from unmask.data.batch import dir_size, free_gb  # noqa: E402
from unmask.data.celebdf import (CelebConfig, extract_celebdf, run_celeb_gates, sample_sheet)  # noqa: E402
from unmask.data.index import read_index  # noqa: E402

OUT = CROPS_DIR / "celebdf_v2_test"
SEARCH = [PROJECT_ROOT / "zip data", Path.home() / "Downloads"]


def find_zip(explicit: Path | None) -> Path:
    if explicit:
        if not explicit.exists():
            raise SystemExit(f"{explicit} not found")
        return explicit
    hits = [p for d in SEARCH if d.exists() for p in d.glob("*.zip") if "celeb" in p.name.lower()]
    if len(hits) != 1:
        raise SystemExit(f"found {len(hits)} Celeb-DF zips in {', '.join(map(str, SEARCH))}: "
                         f"{[str(h) for h in hits]}; pass --zip")
    return hits[0]


def main(argv=None, detector=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="trial: first N listed videos, no DONE.json")
    args = ap.parse_args(argv)
    log, log_path = setup_logger("prepare_celebdf")
    t0, out, trial = time.time(), args.out, args.limit > 0

    def summary(lines, nxt):
        log.info("")
        log.info("==== CELEB-DF SUMMARY ====")
        for line in lines:
            log.info(line)
        log.info(f"crops: {dir_size(out) / 2**20:.0f} MB in {out} | free disk: {free_gb(PROJECT_ROOT):.1f} GB | "
                 f"time {(time.time() - t0) / 60:.1f} min")
        log.info("NEXT: " + nxt)
        log.info(f"log: {log_path}")
        log.info("==========================")

    if (out / "DONE.json").exists() and not trial:
        summary(["DONE.json exists, nothing to do"], "paste this summary back.")
        return
    zp = find_zip(args.zip)
    free = free_gb(PROJECT_ROOT)
    log.info(f"celebdf: zip {zp} ({zp.stat().st_size / 2**30:.2f} GB), free disk {free:.1f} GB")
    if free < 3:
        raise SystemExit("less than 3 GB free; free space first (PLAN rule 10)")
    if detector is None:
        from unmask.data.faces import FaceDetector
        detector = FaceDetector(device="cuda")
    cfg = CelebConfig()
    stats = extract_celebdf(zp, out, cfg, detector, log=log, limit=args.limit)
    rows = read_index(out / "index.csv")
    listed = args.limit or stats["listed"]
    gates = run_celeb_gates(out, rows, listed, stats["flags"])
    for name, ok, msg in gates:
        log.info(f"celebdf: {name} {'PASS' if ok else 'FAIL'}: {msg}")
    sheet = sample_sheet(out, rows)
    passed = all(ok for _, ok, _ in gates)
    count = lambda lab, st: sum(r["label"] == lab and r["status"] == st for r in rows)  # noqa: E731
    lines = [f"listed {stats['listed']} | real ok {count('REAL', 'ok')}, fake ok {count('FAKE', 'ok')} | "
             f"no face {sum(r['status'] == 'no_face' for r in rows)}, read errors "
             f"{sum(r['status'] == 'read_error' for r in rows)}, missing {sum(r['status'] == 'missing' for r in rows)}",
             f"gates {'PASS' if passed else 'FAIL'} | sample sheet {sheet}"]
    if passed and not trial:
        (out / "DONE.json").write_text(json.dumps(dict(
            finished=datetime.now().isoformat(timespec="seconds"), zip=str(zp), listed=stats["listed"],
            gates={n: m for n, _, m in gates}, config=stats["config"], flags=stats["flags"]), indent=1),
            encoding="utf-8")
    summary(lines, "paste this summary back. Do not delete the zip: I will ask first (PLAN section 4)."
            if passed else "paste this summary back; gates failed, nothing is deleted.")
    return gates


if __name__ == "__main__":
    main()
