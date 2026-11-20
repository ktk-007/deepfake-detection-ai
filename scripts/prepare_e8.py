"""E8 (PLAN 10.2): the published GenD checkpoint (yermandy/GenD_CLIP_L_14, trained by its authors on FF++),
converted into this project's checkpoint format without installing `transformers`.

GenD freezes OpenAI's CLIP ViT-L/14 and trains only the LayerNorms and a LinearNorm head, exactly like our E7.
So: download model.safetensors (about 1.2 GB, pre-approved in PLAN 13), check that its frozen backbone equals the
OpenAI weights open_clip already uses (patch embedding, attention and MLP of several blocks), copy its LayerNorm
weights and head into our ClipLN model, and save runs\\e8_gend\\best.pt. If the backbone does not match, nothing
is saved and E8 is reported as not run. Its preprocessing (CLIP mean/std on 224 x 224) is the same as ours.

  python scripts\\prepare_e8.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from unmask.common import PROJECT_ROOT, RUNS_DIR, setup_logger  # noqa: E402
from unmask.data.batch import free_gb  # noqa: E402

REPO = "yermandy/GenD_CLIP_L_14"


def mapping(n_blocks: int) -> dict:
    """our ClipLN name -> GenD (Hugging Face CLIP) name suffix."""
    m = {"visual.ln_pre.weight": "vision_model.pre_layrnorm.weight", "visual.ln_pre.bias": "vision_model.pre_layrnorm.bias",
         "visual.ln_post.weight": "vision_model.post_layernorm.weight", "visual.ln_post.bias": "vision_model.post_layernorm.bias",
         "head.linear.weight": "model.linear.weight", "head.linear.bias": "model.linear.bias"}
    for i in range(n_blocks):
        for ours, theirs in (("ln_1", "layer_norm1"), ("ln_2", "layer_norm2")):
            for p in ("weight", "bias"):
                m[f"visual.transformer.resblocks.{i}.{ours}.{p}"] = f"vision_model.encoder.layers.{i}.{theirs}.{p}"
    return m


def frozen_checks(n_blocks: int) -> dict:
    """our frozen tensor -> GenD tensor that must be identical if GenD kept OpenAI's backbone."""
    c = {"visual.conv1.weight": "vision_model.embeddings.patch_embedding.weight",
         "visual.class_embedding": "vision_model.embeddings.class_embedding"}
    for i in (0, n_blocks // 2, n_blocks - 1):
        c[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"] = f"vision_model.encoder.layers.{i}.mlp.fc1.weight"
        c[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"] = f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
    return c


def find(tensors: dict, suffix: str):
    hits = [k for k in tensors if k.endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"{suffix}: {len(hits)} matches in the GenD checkpoint")
    return tensors[hits[0]]


def main():
    log, log_path = setup_logger("prepare_e8")
    out = RUNS_DIR / "e8_gend" / "best.pt"
    if out.exists():
        log.info(f"{out} exists, nothing to do")
        return out
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    log.info(f"free disk {free_gb(PROJECT_ROOT):.1f} GB; downloading {REPO} model.safetensors (about 1.2 GB)")
    path = hf_hub_download(REPO, "model.safetensors")
    theirs = load_file(path)
    log.info(f"GenD checkpoint: {len(theirs)} tensors")

    from unmask.models.build import build_model
    model = build_model("vit_l14", pretrained=True)
    ours = model.state_dict()
    n_blocks = len(model.visual.transformer.resblocks)

    bad = []
    for o, t in frozen_checks(n_blocks).items():
        a, b = ours[o].float(), find(theirs, t).float().reshape(ours[o].shape)
        if not torch.allclose(a, b, atol=1e-4):
            bad.append(o)
    if bad:
        log.info(f"backbone differs from OpenAI CLIP at {bad}: E8 cannot be converted; report it as not run")
        raise SystemExit(1)
    log.info("frozen backbone matches OpenAI CLIP ViT-L/14 (checked patch embedding, class token, attention and MLP)")

    state = {o: find(theirs, t).float() for o, t in mapping(n_blocks).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys {unexpected[:3]}")
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    if trainable - set(state):
        raise SystemExit(f"trainable tensors not filled: {sorted(trainable - set(state))[:3]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind="best", arch="vit_l14", linear_probe=False, trainable=model.trainable_state_dict(), epoch=0,
                    config=dict(arch="vit_l14", note=f"converted from {REPO} (GenD authors, trained on FF++)"), val={}), out)
    log.info(f"saved {out} ({len(state)} tensors: LayerNorms + LinearNorm head) | log {log_path}")
    return out


if __name__ == "__main__":
    main()
