"""Near-identical score of a fake against its real partner (PLAN 6.2).

noise = mean |fake - real| over the crop border (outside the centre 50%): re-encoding noise of this pair.
local = per frame, the max over 16x16 blocks of the centre 50%; 90th percentile over frames.
score = local / noise. A fake is flagged when score < threshold (2.5): audio-only fakes, or fakes where
the cropped face is not the swapped one.
"""
import numpy as np

THRESHOLD = 2.5


def local_noise_score(fake_crops, real_crops, block: int = 16) -> tuple[float, float]:
    """(score, noise) for two equal-length lists of HxWx3 uint8 crops."""
    a = np.stack(fake_crops).astype(np.int16)
    b = np.stack(real_crops).astype(np.int16)
    d = np.abs(a - b).mean(-1)  # N, H, W
    n, h, w = d.shape
    centre = np.zeros((h, w), bool)
    centre[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    noise = float(d[:, ~centre].mean())
    cen = d[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    ch, cw = cen.shape[1] // block * block, cen.shape[2] // block * block
    blocks = cen[:, :ch, :cw].reshape(n, ch // block, block, cw // block, block).mean((2, 4))
    local = float(np.percentile(blocks.reshape(n, -1).max(1), 90))
    return local / max(noise, 1e-6), noise
