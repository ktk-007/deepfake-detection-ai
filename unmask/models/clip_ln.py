"""CLIP vision tower with LayerNorm tuning (GenD recipe, see docs on the recipe).

Features = ln_post(CLS token) before the visual projection (open_clip: visual.proj = None), 768-d for
ViT-B/16 and 1024-d for ViT-L/14. Trainable: every LayerNorm weight and bias of the vision tower
(ln_pre, ln_1, ln_2 of each block, ln_post) and the linear head. linear_probe=True trains the head only (E1).
"""
import torch
import torch.nn as nn

from .heads import LinearNormHead

ARCHS = {"vit_b16": "ViT-B-16-quickgelu", "vit_l14": "ViT-L-14-quickgelu"}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipLN(nn.Module):
    def __init__(self, arch: str = "vit_b16", pretrained: str | None = "openai", linear_probe: bool = False,
                 grad_checkpointing: bool = False):
        super().__init__()
        import open_clip

        clip = open_clip.create_model(ARCHS[arch], pretrained=pretrained)
        self.visual = clip.visual
        del clip
        self.visual.proj = None
        self.visual.output_tokens = False
        self.dim = self.visual.transformer.width
        self.head = LinearNormHead(self.dim, 2)
        self.arch, self.linear_probe = arch, linear_probe
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

        self.visual.requires_grad_(False)
        if not linear_probe:
            for name, p in self.visual.named_parameters():
                if is_layernorm_param(name):
                    p.requires_grad = True
        if grad_checkpointing:
            self.visual.set_grad_checkpointing(True)

    def forward(self, x: torch.Tensor):
        """x: float images in [0, 1], Bx3x224x224. Returns (logits, z)."""
        x = (x - self.mean) / self.std
        return self.head(self.visual(x))

    def trainable_state_dict(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def param_counts(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train


def is_layernorm_param(name: str) -> bool:
    leaf = name.split(".")
    return any(part in ("ln_pre", "ln_1", "ln_2", "ln_post") for part in leaf)
