"""Save a 4 x 4 grid of augmented training crops per preset (top two rows real, bottom two fake) to logs\\,
so the owner can check by eye that faces stay recognisable (PLAN 9.3). CPU only, a few seconds.

  python scripts\\show_aug.py                      (every preset: none, light, strong, robust, robust_half)
  python scripts\\show_aug.py --preset robust
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from unmask.common import CROPS_DIR, LOG_DIR  # noqa: E402
from unmask.data.augment import PRESETS, build_augment  # noqa: E402
from unmask.data.batch import DEV_PARTS, done_parts  # noqa: E402
from unmask.data.dataset import load_groups, load_image  # noqa: E402



def pick_items(crops_dir: Path, n: int = 8, seed: int = 0):
    """n (real path, fake path) pairs from the development parts, same frame index for both."""
    parts = [p for p in DEV_PARTS if p in done_parts(crops_dir)]
    groups = [g for g in load_groups(parts, crops_dir).values() if g.fakes]
    rng = random.Random(seed)
    out = []
    for g in rng.sample(groups, min(n, len(groups))):
        fdir, fn, _ = rng.choice(g.fakes)
        i = rng.randrange(min(g.n_frames, fn))
        out.append((str(g.real_dir / f"f{i:02d}.jpg"), str(Path(fdir) / f"f{i:02d}.jpg")))
    return out


def tile(img_chw: torch.Tensor, text: str) -> np.ndarray:
    a = cv2.cvtColor(img_chw.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR).copy()
    cv2.rectangle(a, (0, 0), (a.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(a, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return a


def grid(preset: str, pairs, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    aug = build_augment(preset)
    reals = [tile(aug(load_image(r)) if aug else load_image(r), f"REAL {preset}") for r, _ in pairs]
    fakes = [tile(aug(load_image(f)) if aug else load_image(f), f"FAKE {preset}") for _, f in pairs]
    rows = [np.hstack(reals[:4]), np.hstack(reals[4:8]), np.hstack(fakes[:4]), np.hstack(fakes[4:8])]
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, action="append", help="repeatable; default: all presets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    pairs = pick_items(args.crops_dir, 8, args.seed)
    if len(pairs) < 8:
        raise SystemExit("need at least 8 real videos with a usable fake in the development parts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for preset in args.preset or PRESETS:
        path = args.out_dir / f"aug_{preset}.jpg"
        cv2.imwrite(str(path), grid(preset, pairs, args.seed), [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths.append(path)
        print(f"saved {path}")
    return paths


if __name__ == "__main__":
    main()
