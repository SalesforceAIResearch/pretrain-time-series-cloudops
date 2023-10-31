from typing import Optional

import torch
from curated_transformers.layers import QueryKeyRotaryEmbeddings
from gluonts.core.component import validated
from gluonts.torch.util import unsqueeze_expand
from torch import nn, Tensor
from torch.nn import functional as F


class CachedMultiheadAttention(nn.Module):
    @validated()
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        is_decoder: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_decoder = is_decoder

    def _reshape(self, tensor: torch.Tensor):
        return tensor.view(
            tensor.size(0), -1, self.num_heads, self.embed_dim // self.num_heads
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.shape
        query_states = self._reshape(self.q_proj(hidden_states))

        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            key_states, value_states = past_key_value
        elif is_cross_attention:
            key_states = self._reshape(self.k_proj(key_value_states))
            value_states = self._reshape(self.v_proj(key_value_states))
        else:
            key_states = self._reshape(self.k_proj(hidden_states))
            value_states = self._reshape(self.v_proj(hidden_states))

            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        if attn_mask is not None:
            # attn_mask: (bsz, query_len, key_len), insert num_heads dimension
            # F.scaled_dot_product_attention requires an inverted mask
            # True indicates that the element should take part in attention
            attn_mask = unsqueeze_expand(~attn_mask, dim=1, size=self.num_heads)

        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, -1)
        out = self.out_proj(out)

        return out, past_key_value


class MultiheadAttention(nn.Module):
    @validated()
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        rotary_embeds: Optional[QueryKeyRotaryEmbeddings] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.rotary_embeds = rotary_embeds

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        if is_causal and attn_mask is not None:
            raise ValueError("Cannot apply causal mask and attn_mask at the same time.")

        bsz, query_len, _ = query.shape

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query, key, value = map(
            lambda t: t.view(
                bsz, -1, self.num_heads, self.embed_dim // self.num_heads
            ).transpose(1, 2),
            (query, key, value),
        )

        if attn_mask is not None:
            # attn_mask: (bsz, query_len, key_len), insert num_heads dimension
            # F.scaled_dot_product_attention requires an inverted mask
            # True indicates that the element should take part in attention
            attn_mask = unsqueeze_expand(~attn_mask, dim=1, size=self.num_heads)

        if self.rotary_embeds is not None:
            query, key = self.rotary_embeds(
                query=query, key=key, cache=None, positions=None
            )

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, query_len, -1)
        out = self.out_proj(out)
        return out
