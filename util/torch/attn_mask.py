from typing import Optional

import torch
from einops import rearrange
from torch import Tensor

from .ops import block


def causal_mask(
    sz1: int,
    *,
    sz2: Optional[int] = None,
    bsz: tuple[int, ...] = (),
    dtype: torch.dtype = torch.bool,
    device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with True.
    Unmasked positions are filled with False.
    """
    mask = torch.triu(
        block(True, sz1, sz2=sz2, bsz=bsz, dtype=dtype, device=device), diagonal=1
    )
    return mask


def attn_mask(
    observed: Tensor,
    is_causal: bool = False,
    query_length: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> torch.BoolTensor:
    bsz, length = observed.shape[:2]
    query_length = query_length or length

    if observed.ndim > 2:
        observed = observed.max(dim=-1).values

    attn_mask = (
        block(
            False,
            query_length,
            sz2=length,
            bsz=(bsz,),
            device=device,
        )
        + rearrange(
            ~observed.bool(),
            "b l -> b 1 l",
        )
        + (causal_mask(query_length, sz2=length, device=device) if is_causal else False)
    )

    return attn_mask
