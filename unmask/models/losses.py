"""Loss of the GenD recipe (docs: alignment and uniformity of Wang & Isola, 2020).

total = ce_w * CE(logits, y, label_smoothing) + w_align * alignment(z, y) + w_unif * uniformity(z)
z is the L2-normalised embedding. All terms are computed in fp32. Defaults match the GenD repo
(CE 1.0, label smoothing 0.0, alignment 0.1, uniformity 0.5).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def alignment(z: torch.Tensor, y: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Mean ||z_i - z_j||^alpha over same-label pairs i < j. 0 when there is no such pair."""
    if z.size(0) < 2:
        return z.new_zeros(())
    mask = (y[:, None] == y[None, :]).triu(diagonal=1)
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return z.new_zeros(())
    return (z[idx[:, 0]] - z[idx[:, 1]]).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(z: torch.Tensor, t: float = 2.0, clip_value: float = 1e-6) -> torch.Tensor:
    """log(mean(exp(-t ||z_i - z_j||^2))) over all pairs i < j, clamped below at clip_value before the log."""
    if z.size(0) < 2:
        return z.new_zeros(())
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


@dataclass
class LossConfig:
    ce: float = 1.0
    label_smoothing: float = 0.0
    w_align: float = 0.1
    w_unif: float = 0.5


def gend_loss(logits: torch.Tensor, z: torch.Tensor, y: torch.Tensor, cfg: LossConfig = LossConfig()):
    """Returns (total, parts) where parts holds python floats of ce, align, unif."""
    logits, z = logits.float(), z.float()
    ce = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
    total = cfg.ce * ce
    al = alignment(z, y) if cfg.w_align else z.new_zeros(())
    un = uniformity(z) if cfg.w_unif else z.new_zeros(())
    total = total + cfg.w_align * al + cfg.w_unif * un
    return total, {"ce": ce.item(), "align": al.item(), "unif": un.item()}
