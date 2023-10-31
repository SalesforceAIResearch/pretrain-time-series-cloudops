from typing import Callable, Optional

import torch
import torch.nn.functional as F
from curated_transformers.layers import QueryKeyRotaryEmbeddings
from gluonts.core.component import validated
from torch import Tensor, nn

from .attention import MultiheadAttention, CachedMultiheadAttention
from .embeddings import SinusoidalPositionalEmbedding, ScaledQueryKeyRotaryEmbeddings, LearnedPositionalEmbeddings


class TransformerEncoderLayer(nn.Module):
    @validated()
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        rotary_embeds: Optional[QueryKeyRotaryEmbeddings] = None,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, rotary_embeds=rotary_embeds
        )
        self.activation = activation
        self.norm_first = norm_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, src: Tensor, attn_mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], is_causal: bool = False
    ) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    @validated()
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_layers: int = 6,
        norm_first: bool = False,
        max_len: Optional[int] = None,
        interp_len: Optional[int] = None,
        use_sinusoidal_embeds: bool = False,
        use_learned_embeds: bool = False,
        use_rotary_embeds: bool = False,
        use_scaled_rotary_embeds: bool = False
    ):
        super().__init__()
        activation = getattr(F, activation)

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.norm_first = norm_first

        rotary_embeds = None
        self.sinusoidal_embeds = None
        self.learned_embeds = None

        if use_sinusoidal_embeds:
            self.sinusoidal_embeds = SinusoidalPositionalEmbedding(
                width=self.d_model,
                max_len=max_len,
                normalize=False,
                interp_len=interp_len
            )

        if use_learned_embeds:
            self.sinusoidal_embeds = LearnedPositionalEmbeddings(
                width=self.d_model,
                max_len=max_len,
            )

        if use_rotary_embeds:
            rotary_embeds = QueryKeyRotaryEmbeddings(
                fraction=1.0,
                head_width=self.d_model // self.nhead
            )

        if use_scaled_rotary_embeds:
            rotary_embeds = ScaledQueryKeyRotaryEmbeddings(
                fraction=1.0,
                head_width=self.d_model // self.nhead,
                scale=4,
            )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    rotary_embeds=rotary_embeds,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, src: Tensor, attn_mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            raise ValueError(f"attn_mask should be `torch.bool`, not {attn_mask.dtype}")

        output = src

        if self.sinusoidal_embeds is not None:
            output = output + self.sinusoidal_embeds(output.size(1))

        if self.learned_embeds is not None:
            output = output + self.learned_embeds(output.size(1))

        for idx, mod in enumerate(self.layers):
            output = mod(output, attn_mask=attn_mask, is_causal=is_causal)

        return self.norm(output)


class TransformerDecoderLayer(nn.Module):
    @validated()
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.activation = activation
        self.norm_first = norm_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, is_causal=tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], is_causal: bool = False
    ) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, is_causal=is_causal)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoder(nn.Module):
    @validated()
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_layers: int = 6,
        norm_first: bool = False,
    ):
        super().__init__()
        activation = getattr(F, activation)

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.norm_first = norm_first

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
    ) -> Tensor:
        if tgt_mask is not None and tgt_mask.dtype != torch.bool:
            raise ValueError(f"tgt_mask should be `torch.bool`, not {tgt_mask.dtype}")
        if memory_mask is not None and memory_mask.dtype != torch.bool:
            raise ValueError(
                f"memory_mask should be `torch.bool`, not {memory_mask.dtype}"
            )

        output = tgt
        for idx, mod in enumerate(self.layers):
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_is_causal=tgt_is_causal,
            )
        return self.norm(output)


class CachedTransformerDecoderLayer(nn.Module):
    @validated()
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        decoder_only: bool = False,
    ):
        super().__init__()
        self.self_attn = CachedMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, is_decoder=True
        )
        if not decoder_only:
            self.multihead_attn = CachedMultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, is_decoder=True
            )
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.multihead_attn = None
            self.norm2 = None
            self.dropout2 = None
        self.activation = activation
        self.norm_first = norm_first

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        past_key_value: Optional[tuple[Tensor, ...]] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[tuple[Tensor, ...]]]:
        residual = hidden_states

        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        if self.norm_first:
            # Self-attention
            hidden_states, present_key_value = self._sa_block(
                hidden_states=self.norm1(hidden_states),
                attn_mask=tgt_mask,
                past_key_value=self_attn_past_key_value,
            )
            hidden_states = residual + hidden_states

            # Cross-attention
            if memory is not None:
                residual = hidden_states
                cross_attn_past_key_value = (
                    past_key_value[-2:] if past_key_value is not None else None
                )
                hidden_states, cross_attn_present_key_value = self._mha_block(
                    hidden_states=self.norm2(hidden_states),
                    key_value_states=memory,
                    attn_mask=memory_mask,
                    past_key_value=cross_attn_past_key_value,
                )
                hidden_states = residual + hidden_states

                if cross_attn_present_key_value is not None:
                    present_key_value = present_key_value + cross_attn_present_key_value

            # Fully-connected
            hidden_states = hidden_states + self._ff_block(self.norm3(hidden_states))
        else:
            # Self-attention
            hidden_states, present_key_value = self._sa_block(
                hidden_states=hidden_states,
                attn_mask=tgt_mask,
                past_key_value=self_attn_past_key_value,
            )
            hidden_states = self.norm1(residual + hidden_states)

            # Cross-attention
            if memory is not None:
                residual = hidden_states
                cross_attn_past_key_value = (
                    past_key_value[-2:] if past_key_value is not None else None
                )
                hidden_states, cross_attn_present_key_value = self._mha_block(
                    hidden_states=hidden_states,
                    key_value_states=memory,
                    attn_mask=memory_mask,
                    past_key_value=cross_attn_past_key_value,
                )
                hidden_states = self.norm2(residual + hidden_states)
                present_key_value = present_key_value + cross_attn_present_key_value

            # Fully-connected
            hidden_states = hidden_states + self._ff_block(self.norm3(hidden_states))

        if use_cache:
            outputs = (hidden_states, present_key_value)
        else:
            outputs = (hidden_states, None)

        return outputs

    # self-attention block
    def _sa_block(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor],
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attn_mask=attn_mask,
            is_causal=True,
        )
        return self.dropout1(hidden_states), present_key_value

    # multihead attention block
    def _mha_block(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor,
        attn_mask: Optional[Tensor],
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
    ):
        hidden_states, present_key_value = self.multihead_attn(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            attn_mask=attn_mask,
            past_key_value=past_key_value,
        )
        return self.dropout2(hidden_states), present_key_value

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class CachedTransformerDecoder(nn.Module):
    @validated()
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_layers: int = 6,
        norm_first: bool = False,
        decoder_only: bool = False,
    ):
        super().__init__()
        activation = getattr(F, activation)

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_layers = num_layers
        self.norm_first = norm_first

        self.layers = nn.ModuleList(
            [
                CachedTransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    decoder_only=decoder_only,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        past_key_values: Optional[list[tuple[Tensor, ...]]] = None,
        use_cache: bool = True,
    ) -> tuple[Tensor, Optional[list[tuple[Tensor, ...]]]]:
        hidden_states = tgt
        next_decoder_cache = [] if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                tgt_mask=tgt_mask,
                memory=memory,
                memory_mask=memory_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                next_decoder_cache += [present_key_value]

        return self.norm(hidden_states), next_decoder_cache
