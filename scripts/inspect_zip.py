"""Inspect DFDC part zips without extracting: counts, label share, pairing, estimated crop size, free disk.

  python scripts\\inspect_zip.py                 (every dfdc_train_part_NN.zip in "zip data\\")
  python scripts\\inspect_zip.py --zip "zip data\\dfdc_train_part_01.zip"
"""
import argparse
import statistics
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unmask.common import PROJECT_ROOT, ZIP_DIR, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402
from unmask.data.dfdc_zip import ZIP_RE, build_groups, find_part_zips, load_metadata  # noqa: E402
from unmask.data.extract import config_for_part, estimate_mb, planned_videos  # noqa: E402


def inspect(path: Path, log):
    m = ZIP_RE.match(path.name)
    part = int(m.group(1)) if m else -1
    cfg = config_for_part(part)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        meta, prefix, _ = load_metadata(zf)
    reals, fakes_of, info = build_groups(meta, names, prefix)
    n_videos = sum(1 for n in names if n.endswith(".mp4"))
    n_fake = sum(len(v) for v in fakes_of.values())
    counts = [len(v) for v in fakes_of.values()]
    planned = planned_videos(fakes_of, cfg.max_fakes)
    est = estimate_mb(planned, cfg.frames)
    total = info["n_real_meta"] + info["n_fake_meta"]
    log.info(f"part {part:02d}: {path.name}, {path.stat().st_size / 2**30:.2f} GB, {n_videos} mp4 in zip")
    log.info(f"  metadata: {total} videos, real {info['n_real_meta']} ({100 * info['n_real_meta'] / max(1, total):.1f}%), "
             f"fake {info['n_fake_meta']} ({100 * info['n_fake_meta'] / max(1, total):.1f}%)")
    log.info(f"  pairing: {n_fake} fakes have their original in this part; orphans {info['fakes_orphan']}, "
             f"fakes missing from zip {info['fakes_missing']}, reals missing from zip {len(info['reals_missing'])}")
    if counts:
        log.info(f"  fakes per real: min {min(counts)}, max {max(counts)}, mean {statistics.mean(counts):.1f}; "
                 f"reals with no fake {sum(1 for c in counts if c == 0)}")
    log.info(f"  plan: {cfg.frames} frames, up to {cfg.max_fakes} usable fakes per real -> about {planned} videos, "
             f"about {est:.0f} MB of crops (estimate)")
    return dict(part=part, reals=len(reals), fakes=n_fake, planned=planned, est_mb=est)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="append", help="zip path (repeatable); default: all in zip data")
    args = ap.parse_args()
    log, log_path = setup_logger("inspect_zip")
    paths = [Path(z) if Path(z).is_absolute() else PROJECT_ROOT / z for z in args.zip] if args.zip \
        else list(find_part_zips(ZIP_DIR).values())
    if not paths:
        log.info(f'no dfdc_train_part_NN.zip found in "{ZIP_DIR}"')
        return
    res = [inspect(p, log) for p in paths]
    log.info("")
    log.info("==== ZIP SUMMARY ====")
    for r in res:
        log.info(f"part {r['part']:02d}: reals {r['reals']}, fakes {r['fakes']}, planned videos {r['planned']}, "
                 f"crops about {r['est_mb']:.0f} MB")
    log.info(f"crops total about {sum(r['est_mb'] for r in res) / 1024:.2f} GB | free disk {free_gb(PROJECT_ROOT):.1f} GB")
    log.info(f"log: {log_path}")
    log.info("=====================")


if __name__ == "__main__":
    main()
