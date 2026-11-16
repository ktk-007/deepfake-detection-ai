"""Training augmentation presets (torchvision.transforms.v2 only). Applied identically to both classes.

light  = the GenD repo's augmentation set exactly: HFlip 0.5; RandomAffine(10 deg, translate 0.1,
         scale 0.9-1.1); GaussianBlur(k=7, sigma 0.1-2.0) with p=0.1; ColorJitter(brightness 0.1, contrast 0.1);
         JPEG quality 40-100.
strong = light with stronger colour jitter (+ saturation, hue), JPEG 30-100 and GridMask-style dropout.
robust = strong with dropout at p 0.3, plus crop-zoom (p 0.5, 1.0-1.35x), downscale-upscale (p 0.3, 0.4-1.0,
         random interpolations), Gaussian noise (p 0.2, sigma 2-8) and a second JPEG pass (p 0.2, 50-90). Aimed
         at tighter crops and lower video quality in the outside test sets (PLAN 9.3).
none   = no augmentation.
Input and output: uint8 tensors 3xHxW.
"""
import cv2
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF


class GridDropout(torch.nn.Module):
    """GridMask-style: with probability p, zero square patches on a grid with a random period and offset."""

    def __init__(self, p: float = 0.5, period=(32, 96), ratio=(0.3, 0.5)):
        super().__init__()
        self.p, self.period, self.ratio = p, period, ratio

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return img
        _, h, w = img.shape
        d = int(torch.randint(self.period[0], self.period[1] + 1, ()))
        s = max(1, int(d * float(torch.empty(()).uniform_(*self.ratio))))
        ox, oy = int(torch.randint(0, d, ())), int(torch.randint(0, d, ()))
        mask = torch.ones(h, w, dtype=torch.bool)
        for y in range(oy - d, h, d):
            for x in range(ox - d, w, d):
                mask[max(0, y):max(0, y + s), max(0, x):max(0, x + s)] = False
        return img * mask.to(img.dtype)


def _u(lo: float, hi: float) -> float:
    return float(torch.empty(()).uniform_(lo, hi))


class CropZoom(torch.nn.Module):
    """With probability p, zoom in about the centre by a factor in `zoom` (1.3 is about a zero-margin crop of our
    30%-margin face crops), shift the centre by up to `shift` x side, resize back to the input size."""

    def __init__(self, p: float = 0.5, zoom=(1.0, 1.35), shift: float = 0.04):
        super().__init__()
        self.p, self.zoom, self.shift = p, zoom, shift

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return img
        _, h, w = img.shape
        f = _u(*self.zoom)
        ch, cw = max(8, round(h / f)), max(8, round(w / f))
        cy = h / 2 + _u(-self.shift, self.shift) * h
        cx = w / 2 + _u(-self.shift, self.shift) * w
        top = int(round(min(max(cy - ch / 2, 0), h - ch)))
        left = int(round(min(max(cx - cw / 2, 0), w - cw)))
        return TF.resized_crop(img, top, left, ch, cw, [h, w], antialias=True)


_INTERPS = (cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST)


class DownUp(torch.nn.Module):
    """With probability p, resize to a factor in `scale` of the side and back, with an interpolation drawn
    from bilinear, bicubic, area and nearest for each direction (low-resolution, re-encoded video)."""

    def __init__(self, p: float = 0.3, scale=(0.4, 1.0)):
        super().__init__()
        self.p, self.scale = p, scale

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return img
        _, h, w = img.shape
        s = _u(*self.scale)
        a = img.permute(1, 2, 0).contiguous().numpy()
        i1, i2 = (_INTERPS[int(torch.randint(0, len(_INTERPS), ()))] for _ in range(2))
        small = cv2.resize(a, (max(8, round(w * s)), max(8, round(h * s))), interpolation=i1)
        back = cv2.resize(small, (w, h), interpolation=i2)
        return torch.from_numpy(back).permute(2, 0, 1).contiguous()


class GaussNoise(torch.nn.Module):
    """With probability p, add Gaussian noise with sigma in `sigma` on the 0-255 scale."""

    def __init__(self, p: float = 0.2, sigma=(2.0, 8.0)):
        super().__init__()
        self.p, self.sigma = p, sigma

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return img
        noisy = img.float() + torch.randn(img.shape) * _u(*self.sigma)
        return noisy.round_().clamp_(0, 255).to(torch.uint8)


def build_augment(preset: str = "light"):
    if preset == "none":
        return None
    common = [
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.1),
    ]
    if preset == "light":
        return T.Compose(common + [T.ColorJitter(brightness=0.1, contrast=0.1), T.JPEG((40, 100))])
    if preset == "strong":
        return T.Compose(common + [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.JPEG((30, 100)),
            GridDropout(p=0.5),
        ])
    if preset == "robust":
        # PLAN 9.3 order: flip, affine, crop-zoom, colour, blur, downscale, noise, JPEG, second JPEG, dropout.
        return T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            CropZoom(p=0.5, zoom=(1.0, 1.35), shift=0.04),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.1),
            DownUp(p=0.3, scale=(0.4, 1.0)),
            GaussNoise(p=0.2, sigma=(2.0, 8.0)),
            T.JPEG((30, 100)),
            T.RandomApply([T.JPEG((50, 90))], p=0.2),
            GridDropout(p=0.3),
        ])
    raise ValueError(f"unknown augmentation preset {preset!r}")
