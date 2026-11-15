"""Crop index, paired sampler and datasets (PLAN 8.2).

Groups: one real video (status ok) and its fakes. Usable fakes = status ok and flagged == 0 (or all fakes
with include_flagged). One epoch = every training group once: the real and one usable fake (rotating across
epochs), the same `frames_per_video` random frame indices for both. A batch holds whole pairs, so it is always
50% real, 50% fake. A real with no usable fake is paired with a usable fake of another real of the same part.
paired=False (ablation): the fake always comes from another original of the same part.

The batch sampler yields lists of items (path, label, video_key); the dataset loads an item directly, so
persistent DataLoader workers never need the per-epoch plan.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, Sampler

from ..common import CROPS_DIR
from .index import read_index, video_dir
from .splits import group_key


@dataclass
class Group:
    key: str
    part: int
    real_dir: Path
    n_frames: int
    fakes: list = field(default_factory=list)  # list of (dir, n_frames, video_key)


def load_groups(parts, crops_dir: Path = CROPS_DIR, include_flagged: bool = False) -> dict[str, Group]:
    groups = {}
    for part in parts:
        d = crops_dir / f"part_{int(part):02d}"
        rows = read_index(d / "index.csv")
        for r in rows:
            if r["label"] == "REAL" and r["status"] == "ok":
                k = group_key(part, r["video"])
                groups[k] = Group(k, int(part), video_dir(d, r["video"]), int(r["n_frames"]))
        for r in rows:
            if r["label"] != "FAKE" or r["status"] != "ok":
                continue
            if not include_flagged and str(r["flagged"]) == "1":
                continue
            k = group_key(part, r["original"])
            if k in groups:
                groups[k].fakes.append((video_dir(d, r["video"]), int(r["n_frames"]), group_key(part, r["video"])))
    return groups


class PairedBatchSampler(Sampler):
    def __init__(self, groups: list[Group], pairs_per_batch: int, frames_per_video: int = 4, paired: bool = True,
                 seed: int = 0, drop_last: bool = True):
        self.groups = sorted(groups, key=lambda g: g.key)
        self.ppb, self.fpv, self.paired, self.seed, self.drop_last = pairs_per_batch, frames_per_video, paired, seed, drop_last
        self.epoch = 0
        self.by_part: dict[int, list] = {}
        for g in self.groups:
            for f in g.fakes:
                self.by_part.setdefault(g.part, []).append((g.key, f))
        if not any(self.by_part.values()):
            raise ValueError("no usable fakes in the training groups")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        n = len(self.groups)
        return n // self.ppb if self.drop_last else -(-n // self.ppb)

    def _pick_fake(self, g: Group, rng: random.Random):
        if self.paired and g.fakes:
            return g.fakes[(self.epoch + _stable_offset(g.key)) % len(g.fakes)]
        pool = [f for k, f in self.by_part.get(g.part, []) if k != g.key] or \
               [f for lst in self.by_part.values() for k, f in lst if k != g.key]
        return rng.choice(pool)

    def epoch_pairs(self) -> list[list[tuple]]:
        rng = random.Random(self.seed * 100003 + self.epoch)
        order = list(self.groups)
        rng.shuffle(order)
        pairs = []
        for g in order:
            fdir, fn, fkey = self._pick_fake(g, rng)
            n = min(g.n_frames, fn)
            idx = sorted(rng.sample(range(n), min(self.fpv, n)))
            items = [(str(g.real_dir / f"f{i:02d}.jpg"), 0, g.key) for i in idx]
            items += [(str(fdir / f"f{i:02d}.jpg"), 1, fkey) for i in idx]
            pairs.append(items)
        return pairs

    def __iter__(self):
        pairs = self.epoch_pairs()
        for b in range(len(self)):
            chunk = pairs[b * self.ppb:(b + 1) * self.ppb]
            yield [it for p in chunk for it in p]


def _stable_offset(key: str) -> int:
    return sum(key.encode()) % 997


def load_image(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot read {path}")
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()


class ItemDataset(Dataset):
    """Loads an item (path, label, video_key) -> (float image in [0,1], label, video_key)."""

    def __init__(self, augment=None):
        self.augment = augment

    def __getitem__(self, item):
        path, label, key = item
        img = load_image(path)
        if self.augment is not None:
            img = self.augment(img)
        return img.float().div_(255), label, key


class EvalFrames(Dataset):
    """Every stored frame of every real and usable fake of the given groups (no augmentation)."""

    def __init__(self, groups: list[Group]):
        self.items = []
        for g in sorted(groups, key=lambda g: g.key):
            self.items += [(str(g.real_dir / f"f{i:02d}.jpg"), 0, g.key) for i in range(g.n_frames)]
            for fdir, fn, fkey in g.fakes:
                self.items += [(str(fdir / f"f{i:02d}.jpg"), 1, fkey) for i in range(fn)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, key = self.items[i]
        return load_image(path).float().div_(255), label, key


def collate(batch):
    imgs, labels, keys = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), list(keys)
