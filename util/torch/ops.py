from typing import Optional

import torch
from torch import Tensor


def unsqueeze_dim(x: Tensor, shape: torch.Size) -> Tensor:
    dim = (...,) + (None,) * len(shape)
    return x[dim]


def block(
    value: bool,
    sz1: int,
    *,
    sz2: Optional[int] = None,
    bsz: tuple[int, ...] = (),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bool,
) -> Tensor:
    shape = (sz1, sz2) if sz2 is not None else (sz1, sz1)
    return (torch.ones if value else torch.zeros)(
        bsz + shape, dtype=dtype, device=device
    )
