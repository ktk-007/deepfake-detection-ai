"""Build any model of the project by name. Every model maps float images in [0, 1] (Bx3x224x224) to
(logits Bx2, z) where z is the L2-normalised embedding (None for EfficientNet).

  vit_b16 / vit_l14   CLIP vision tower, LayerNorm + head trainable (linear_probe=True: head only, E1)
  effb4               timm tf_efficientnet_b4.ns_jft_in1k, all layers trainable (E3 baseline)
  tiny                small CNN for tests (no download)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import LinearNormHead

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per-architecture training defaults (PLAN 9.4). batch = images per step = pairs x frames x 2.
ARCH_DEFAULTS = {
    "vit_b16": dict(batch=64, lr=3e-4, weight_decay=0.0, grad_ckpt=False, loss="gend", label_smoothing=0.0),
    "vit_l14": dict(batch=32, lr=3e-4, weight_decay=0.0, grad_ckpt=True, loss="gend", label_smoothing=0.0),
    "effb4": dict(batch=32, lr=1e-4, weight_decay=1e-2, grad_ckpt=False, loss="ce", label_smoothing=0.01),
    "tiny": dict(batch=8, lr=1e-3, weight_decay=0.0, grad_ckpt=False, loss="gend", label_smoothing=0.0),
}


class _Normalised(nn.Module):
    """Shared helpers: input normalisation and trainable-parameter bookkeeping."""

    def _set_norm(self, mean, std):
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        sd = self.state_dict()
        # BatchNorm running stats are buffers, not parameters, but they change in training: keep them too.
        bufs = {n for n, _ in self.named_buffers() if n.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        return {k: v.detach().cpu().clone() for k, v in sd.items() if k in names or k in bufs}

    def param_counts(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train


class EffNetB4(_Normalised):
    def __init__(self, pretrained: bool = True, grad_checkpointing: bool = False):
        super().__init__()
        import timm

        self.net = timm.create_model("tf_efficientnet_b4.ns_jft_in1k", pretrained=pretrained, num_classes=2)
        if grad_checkpointing:
            self.net.set_grad_checkpointing(True)
        self._set_norm(IMAGENET_MEAN, IMAGENET_STD)
        self.arch, self.linear_probe = "effb4", False

    def forward(self, x):
        return self.net((x - self.mean) / self.std), None


class TinyNet(_Normalised):
    """A few conv layers + LinearNorm head; for unit tests only."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 5, 4), nn.ReLU(), nn.Conv2d(8, 16, 3, 2), nn.ReLU(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = LinearNormHead(16, 2)
        self._set_norm(IMAGENET_MEAN, IMAGENET_STD)
        self.arch, self.linear_probe = "tiny", False

    def forward(self, x):
        return self.head(self.body((x - self.mean) / self.std))


def build_model(arch: str, linear_probe: bool = False, grad_ckpt: bool = False, pretrained: bool = True):
    if arch in ("vit_b16", "vit_l14"):
        from .clip_ln import ClipLN

        return ClipLN(arch, pretrained="openai" if pretrained else None, linear_probe=linear_probe,
                      grad_checkpointing=grad_ckpt)
    if linear_probe:
        raise ValueError("linear_probe applies to the CLIP models only")
    if arch == "effb4":
        return EffNetB4(pretrained=pretrained, grad_checkpointing=grad_ckpt)
    if arch == "tiny":
        return TinyNet()
    raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCH_DEFAULTS)}")


def check_quickgelu(model) -> bool | None:
    """True if every CLIP block uses QuickGELU (OpenAI weights need it). None for non-CLIP models."""
    vis = getattr(model, "visual", None)
    if vis is None:
        return None
    acts = [type(m).__name__ for m in vis.modules() if "GELU" in type(m).__name__]
    return bool(acts) and all(a == "QuickGELU" for a in acts)


def frame_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=1)[:, 1]
