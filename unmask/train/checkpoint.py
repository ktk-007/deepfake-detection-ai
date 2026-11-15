"""Small checkpoints for CLIP models: only the trainable tensors (LayerNorms + head) plus the config."""
from pathlib import Path

import torch

from ..models.clip_ln import ClipLN


def save_clip_checkpoint(model: ClipLN, path: Path, config: dict, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=config, arch=model.arch, linear_probe=model.linear_probe,
                    trainable=model.trainable_state_dict(), **(extra or {})), path)


def load_clip_checkpoint(path: Path, device="cpu", pretrained="openai") -> tuple[ClipLN, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ClipLN(ck["arch"], pretrained=pretrained, linear_probe=ck["linear_probe"])
    missing, unexpected = model.load_state_dict(ck["trainable"], strict=False)
    if unexpected:
        raise KeyError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    not_loaded = {n for n, p in model.named_parameters() if p.requires_grad} - set(ck["trainable"])
    if not_loaded:
        raise KeyError(f"trainable parameters missing from checkpoint: {sorted(not_loaded)[:5]}")
    return model.to(device), ck
