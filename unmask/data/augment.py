"""Training augmentation presets (torchvision.transforms.v2 only). Applied identically to both classes.

light  = the GenD repo's augmentation set exactly: HFlip 0.5; RandomAffine(10 deg, translate 0.1,
         scale 0.9-1.1); GaussianBlur(k=7, sigma 0.1-2.0) with p=0.1; ColorJitter(brightness 0.1, contrast 0.1);
         JPEG quality 40-100.
strong = light with stronger colour jitter (+ saturation, hue), JPEG 30-100 and GridMask-style dropout.
none   = no augmentation.
Input and output: uint8 tensors 3xHxW.
"""
import torch
from torchvision.transforms import v2 as T


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
    raise ValueError(f"unknown augmentation preset {preset!r}")
