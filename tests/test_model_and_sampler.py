import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from unmask.data.dataset import Group, PairedBatchSampler
from unmask.eval.metrics import binary_metrics, video_scores
from unmask.models.losses import LossConfig, alignment, gend_loss, uniformity


# ---------------------------------------------------------------- losses
def test_alignment_and_uniformity_match_brute_force():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(10, 8), dim=1)
    y = torch.tensor([0, 1] * 5)
    pairs = [(i, j) for i, j in itertools.combinations(range(10), 2) if y[i] == y[j]]
    want_al = np.mean([float((z[i] - z[j]).norm() ** 2) for i, j in pairs])
    assert alignment(z, y).item() == pytest.approx(want_al, rel=1e-5)
    allp = list(itertools.combinations(range(10), 2))
    want_un = np.log(np.mean([np.exp(-2 * float((z[i] - z[j]).norm() ** 2)) for i, j in allp]))
    assert uniformity(z).item() == pytest.approx(want_un, rel=1e-5)


def test_alignment_zero_without_positive_pairs_and_total_loss():
    z = F.normalize(torch.randn(2, 4), dim=1)
    assert alignment(z, torch.tensor([0, 1])).item() == 0
    logits = torch.randn(6, 2, requires_grad=True)
    zz = F.normalize(torch.randn(6, 4), dim=1)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    total, parts = gend_loss(logits, zz, y, LossConfig())
    expect = F.cross_entropy(logits, y) + 0.1 * alignment(zz, y) + 0.5 * uniformity(zz)
    assert total.item() == pytest.approx(expect.item(), rel=1e-6)
    assert set(parts) == {"ce", "align", "unif"}


# ---------------------------------------------------------------- sampler
def _groups():
    gs = []
    for part in (0, 1):
        for i in range(6):
            fakes = [(Path(f"p{part}/f{i}_{k}"), 16, f"{part:02d}/f{i}_{k}.mp4") for k in range(3 if i else 0)]
            gs.append(Group(f"{part:02d}/r{i}.mp4", part, Path(f"p{part}/r{i}"), 16, fakes))
    return gs


def test_paired_batches_are_balanced_pairs_with_same_frames():
    s = PairedBatchSampler(_groups(), pairs_per_batch=4, frames_per_video=4, seed=0)
    batches = list(s)
    assert len(batches) == 3 and all(len(b) == 32 for b in batches)
    for b in batches:
        assert Counter(lab for _, lab, _ in b) == {0: 16, 1: 16}
        for p in range(4):
            pair = b[p * 8:(p + 1) * 8]
            real, fake = pair[:4], pair[4:]
            assert [Path(x[0]).name for x in real] == [Path(x[0]).name for x in fake]
            assert all(x[1] == 0 for x in real) and all(x[1] == 1 for x in fake)
    seen = [x[2] for b in batches for x in b if x[1] == 0]
    assert len(set(seen)) == 12  # every training real once per epoch


def test_fakes_rotate_and_fakeless_real_uses_same_part_fake():
    s = PairedBatchSampler(_groups(), pairs_per_batch=1, frames_per_video=2, seed=0)
    used = {}
    for e in range(3):
        s.set_epoch(e)
        for pair in s.epoch_pairs():
            real_key = pair[0][2]
            fake_path = pair[-1][0]
            used.setdefault(real_key, set()).add(str(Path(fake_path).parent))
            part = real_key[:2]
            assert Path(fake_path).parts[0] == f"p{int(part)}"
    assert all(len(v) == 3 for k, v in used.items() if not k.endswith("/r0.mp4"))


def test_unpaired_mode_never_uses_own_fakes():
    s = PairedBatchSampler(_groups(), pairs_per_batch=1, frames_per_video=2, paired=False, seed=0)
    for pair in s.epoch_pairs():
        real_part, real_name = Path(pair[0][0]).parts[:2]        # ("p0", "r3")
        fake_part, fake_name = Path(pair[-1][0]).parts[:2]       # ("p0", "f5_1")
        own_fake = fake_part == real_part and fake_name.split("_")[0][1:] == real_name[1:]
        assert not own_fake


# ---------------------------------------------------------------- metrics
def test_video_scores_and_metrics():
    keys = ["a", "a", "b", "b", "c", "c"]
    vids, y, s = video_scores([0.1, 0.3, 0.9, 0.7, 0.4, 0.6], [0, 0, 1, 1, 1, 1], keys)
    assert vids == ["a", "b", "c"] and s.tolist() == pytest.approx([0.2, 0.8, 0.5])
    m = binary_metrics(y, s)
    assert m["auc"] == 1.0 and m["n_real"] == 1 and m["n_fake"] == 2


# ---------------------------------------------------------------- model wiring (no weights download)
def test_clip_ln_trains_only_layernorms_and_head():
    from unmask.models.clip_ln import ClipLN

    m = ClipLN("vit_b16", pretrained=None)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert all(("ln_" in n) or n.startswith("head.") for n in names)
    assert "visual.ln_pre.weight" in names and "visual.ln_post.bias" in names and "head.linear.weight" in names
    assert sum("ln_1" in n for n in names) == 24  # 12 blocks x (weight, bias)
    total, train = m.param_counts()
    assert 0.0002 < train / total < 0.0006
    logits, z = m(torch.rand(2, 3, 224, 224))
    assert logits.shape == (2, 2) and z.shape == (2, 768)
    assert torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5)
    probe = ClipLN("vit_b16", pretrained=None, linear_probe=True)
    assert all(n.startswith("head.") for n, p in probe.named_parameters() if p.requires_grad)
