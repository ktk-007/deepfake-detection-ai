import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNormHead(nn.Module):
    """GenD 'LinearNorm' head: z = x / ||x||, logits = Linear(z). Returns (logits, z)."""

    def __init__(self, dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor):
        z = F.normalize(x.float(), p=2, dim=1)
        return self.linear(z.to(self.linear.weight.dtype)), z
