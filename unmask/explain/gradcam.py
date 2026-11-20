"""Grad-CAM style maps: "areas that influenced the result" (PLAN 11), never proof of manipulation.

CLIP ViT: gradients of the fake logit with respect to the patch tokens entering several of the later transformer
blocks (14 x 14 grid for ViT-B/16, 16 x 16 for ViT-L/14). The last block's own patch outputs are not used: the
classifier reads only the CLS token after it, so their gradient is zero. EfficientNet-B4: the last conv feature map.
For each layer: channel weights = mean gradient, map = ReLU(weighted sum), normalised; layers are averaged and the
whole is averaged over a few noisy copies of the input (SmoothGrad), which removes most single-patch speckle.
The overlay keeps only the face region (a face swap changes the face, and ViT maps otherwise light up background
tokens), colours only the strongest areas, and scales the colour by the frame's fake probability, so a frame the
model finds real shows little or no heat. The model weights are frozen; no parameter is changed.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def target_layers(model) -> list:
    if hasattr(model, "visual"):                       # ClipLN: later blocks, never the last one
        blocks = model.visual.transformer.resblocks
        n = len(blocks)
        idx = sorted({min(round(n * f), n - 2) for f in (0.67, 0.75, 0.83, 0.92)})
        return [blocks[i] for i in idx]
    if hasattr(model, "net") and hasattr(model.net, "conv_head"):  # EffNetB4 (timm)
        return [model.net.conv_head]
    if hasattr(model, "body"):                          # TinyNet (tests)
        return [model.body[2]]
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")


def _layer_cam(act: torch.Tensor, grad: torch.Tensor, batch: int) -> torch.Tensor:
    act, grad = act.detach().float(), grad.detach().float()
    if act.dim() == 3:                                  # ViT tokens, batch-first or sequence-first
        if act.shape[0] != batch:
            act, grad = act.permute(1, 0, 2), grad.permute(1, 0, 2)
        act, grad = act[:, 1:], grad[:, 1:]             # drop the CLS token
        side = int(round(act.shape[1] ** 0.5))
        cam = F.relu((grad.mean(dim=1, keepdim=True) * act).sum(-1)).view(batch, 1, side, side)
    else:                                               # conv feature map B x C x h x w
        cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(1, keepdim=True))
    return cam / cam.flatten(1).max(1)[0].view(-1, 1, 1, 1).clamp_min(1e-8)


def gradcam(model, x: torch.Tensor, size: int = 224, passes: int = 8, noise: float = 0.08, seed: int = 0) -> np.ndarray:
    """x: float images in [0, 1], Bx3xHxW on the model's device. Returns B x size x size maps in [0, 1]."""
    layers = target_layers(model)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    total = None
    for k in range(max(1, passes)):
        store = {}

        def make_hook(j):
            def hook(_m, _inp, out):
                out = out[0] if isinstance(out, tuple) else out
                if not out.requires_grad:
                    out = out.detach().requires_grad_(True)
                out.retain_grad()
                store[j] = out
                return out
            return hook

        hooks = [layer.register_forward_hook(make_hook(j)) for j, layer in enumerate(layers)]
        try:
            xi = x.float()
            if k > 0 and noise > 0:
                xi = (xi + noise * torch.randn(xi.shape, generator=gen, device=xi.device)).clamp(0, 1)
            with torch.enable_grad():
                logits, _ = model(xi)
                model.zero_grad(set_to_none=True)
                logits[:, 1].sum().backward()
        finally:
            for h in hooks:
                h.remove()
        cams = [_layer_cam(store[j], store[j].grad, x.shape[0]) for j in range(len(layers))]
        cams = [F.interpolate(c, size=(size, size), mode="bilinear", align_corners=False) for c in cams]
        step = torch.stack(cams).mean(0)
        total = step if total is None else total + step
    cam = total[:, 0].cpu().numpy()
    out = []
    for c in cam:
        c = cv2.GaussianBlur(c, (0, 0), size / 28)
        out.append((c - c.min()) / max(float(c.max() - c.min()), 1e-8))
    return np.stack(out)


def face_mask(size: int = 224) -> np.ndarray:
    """Soft ellipse over the face region of our crops (30% margin around the detected face)."""
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    return np.clip(1.4 - np.sqrt(((xx - 0.5) / 0.34) ** 2 + ((yy - 0.53) / 0.42) ** 2), 0, 1) ** 0.7


def overlay(crop_bgr: np.ndarray, cam: np.ndarray, strength: float = 1.0, max_regions: int = 2) -> np.ndarray:
    """Red tint on the strongest face areas plus a white outline around the (at most two) largest key regions.
    `strength` is the frame's fake probability: the marking fades out below about 0.3, so frames the model finds
    real stay unmarked. Checked against occlusion sensitivity (hiding an area and measuring the score drop): both
    point at the same regions on the demo fakes."""
    k = cam * face_mask(cam.shape[0])
    k = (k - k.min()) / max(float(k.max() - k.min()), 1e-8)
    s = float(np.clip((strength - 0.3) / 0.5, 0.0, 1.0))
    out = crop_bgr.astype(np.float32)
    a = (np.clip((k - 0.45) / 0.55, 0, 1) * 0.55 * s)[..., None]
    red = np.zeros_like(out)
    red[..., 2], red[..., 1] = 255, 40 * k
    out = out * (1 - a) + red * a
    if s > 0.05:
        contours, _ = cv2.findContours(np.uint8(k > 0.6), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.006 * k.size
        contours = sorted([c for c in contours if cv2.contourArea(c) >= min_area], key=cv2.contourArea, reverse=True)
        contours = contours[:max_regions]
        cv2.drawContours(out, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, (40, 40, 220), 1, cv2.LINE_AA)
    return np.uint8(np.clip(out, 0, 255))
