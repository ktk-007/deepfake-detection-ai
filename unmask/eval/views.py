"""Deterministic stress views of the part-08 validation crops (PLAN 9.5).

clean     nothing
tight     centre zoom by 1.3, bicubic resize back to the input size (about a zero-margin crop)
degraded  bilinear resize to half the side and back, then JPEG quality 50

They imitate the outside test sets (tighter crops, lower quality) using validation data only.
Input and output: uint8 tensors 3xHxW.
"""
import cv2
import numpy as np
import torch

VIEWS = ("clean", "tight", "degraded")


def _np(img: torch.Tensor) -> np.ndarray:
    return img.permute(1, 2, 0).contiguous().numpy()


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).contiguous()


def tight(img: torch.Tensor, zoom: float = 1.3) -> torch.Tensor:
    _, h, w = img.shape
    ch, cw = round(h / zoom), round(w / zoom)
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = _np(img)[top:top + ch, left:left + cw]
    return _t(cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC))


def degraded(img: torch.Tensor, quality: int = 50) -> torch.Tensor:
    _, h, w = img.shape
    a = _np(img)
    a = cv2.resize(cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR), (w, h),
                   interpolation=cv2.INTER_LINEAR)
    # images are RGB here; JPEG round trip is colour-order agnostic as long as encode and decode match
    ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return _t(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED))


def view_fn(name: str):
    """None for clean (no transform), else the view function."""
    if name == "clean":
        return None
    if name == "tight":
        return tight
    if name == "degraded":
        return degraded
    raise ValueError(f"unknown view {name!r}; choose from {VIEWS}")
