import math
from typing import Optional

import torch
from curated_transformers.layers import RotaryEmbeddings, QueryKeyRotaryEmbeddings
from torch import nn, Tensor


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        *,
        width: int,
        max_len: int,
        normalize=True,
        device: Optional[torch.device] = None,
        interp_len: Optional[int] = None,
    ):
        """
        Construct a sinusoidal positional embedding module.

        :param width:
            Width of the embedding.
        :param max_len:
            Maximum length of the embedding.
        :param normalize:
            Perform L2 normalization of the embedding.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        if interp_len is not None:
            position = torch.linspace(0, interp_len - 1, max_len, device=device).unsqueeze(1)
        else:
            position = torch.arange(max_len, device=device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, width, 2, device=device) * (-math.log(10000.0) / width)
        )

        pe = torch.zeros(max_len, width, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if normalize == True:
            l2 = torch.linalg.vector_norm(pe, dim=-1)
            pe /= l2.unsqueeze(-1)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int) -> Tensor:
        """
        Returns the positional embedding for the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Positional embedding for the input.

            *Shape:* ``(seq_len, width)``
        """
        return self.pe[: length, :]


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(
            self,
            *,
            width: int,
            max_len: int,
    ):
        super().__init__()
        self.pe = nn.Embedding(max_len, width, )
        self.register_buffer(
            "position_ids", torch.arange(max_len).unsqueeze(0), persistent=False
        )

    def forward(self, length: int) -> Tensor:
        """
        Returns the positional embedding for the input.

        :param input:
            Input tensor.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Positional embedding for the input.

            *Shape:* ``(seq_len, width)``
        """
        return self.pe(self.position_ids[:, :length])


class ScaledRotaryEmbedding(RotaryEmbeddings):
    def __init__(
        self,
        width: int,
        *,
        scale: int = 4,
        seq_len: int = 512,
        base: int = 10000,
        device: Optional[torch.device] = None,
    ):
        self.scale = scale
        super().__init__(width=width, seq_len=seq_len, base=base, device=device)

    def _create_rotary_embed(self, *, width: int, length: int):
        # mΘ
        position = torch.arange(length, device=self.theta.device).unsqueeze(1) / self.scale
        m_theta = position * self.theta.unsqueeze(0)

        # We apply both sin and cos twice (see Eq 15, 34), but the ordering
        # is changed for compatibility with most common implementations.
        m_theta = torch.cat([m_theta, m_theta], dim=-1)

        re_cos = m_theta.cos().view([length, width])
        re_sin = m_theta.sin().view([length, width])

        self.register_buffer("cos", re_cos, persistent=False)
        self.register_buffer("sin", re_sin, persistent=False)


class ScaledQueryKeyRotaryEmbeddings(QueryKeyRotaryEmbeddings):
    def __init__(
        self,
        *,
        base: int = 10000,
        fraction: float,
        head_width: int,
        scale: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(base=base, fraction=fraction, head_width=head_width, device=device)
        self.rotary_embeds = ScaledRotaryEmbedding(
            width=self.rotary_width, scale=scale, base=base, device=device
        )
