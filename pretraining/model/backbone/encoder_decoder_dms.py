from functools import cached_property
from typing import Optional

import torch
from einops import rearrange
from gluonts.itertools import prod
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    unsqueeze_expand,
    weighted_average,
)
from torch import nn, Tensor

from pretraining.model.backbone.layers.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from util.torch.scaler import StdScaler, NOPScaler
from util.torch.attn_mask import attn_mask


class EncoderDecoderDMSModel(nn.Module):
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        time_dim: int,
        static_dim: int,
        dynamic_dim: int,
        past_dynamic_dim: int,
        static_cardinalities: list[int],
        dynamic_cardinalities: list[int],
        past_dynamic_cardinalities: list[int],
        static_embedding_dim: list[int],
        dynamic_embedding_dim: list[int],
        past_dynamic_embedding_dim: list[int],
        lags_seq: list[int],
        scaling: bool = True,
        distr_output: DistributionOutput = StudentTOutput(),
        num_parallel_samples: int = 100,
        # Model args
        d_model: int = 32,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_dim = time_dim
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.past_dynamic_dim = past_dynamic_dim
        self.static_cardinalities = static_cardinalities
        self.dynamic_cardinalities = dynamic_cardinalities
        self.past_dynamic_cardinalities = past_dynamic_cardinalities
        self.static_embedding_dim = static_embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        self.past_dynamic_embedding_dim = past_dynamic_embedding_dim
        self.lags_seq = lags_seq
        self.num_parallel_samples = num_parallel_samples

        self.scaling = scaling
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout

        # Output distribution
        self.distr_output = distr_output
        self.out_proj = distr_output.get_args_proj(d_model)
        self.target_shape = distr_output.event_shape
        self.target_dim = prod(self.target_shape)

        # Scaling
        self.scaler = (
            StdScaler(dim=1, keepdim=True)
            if scaling
            else NOPScaler(dim=1, keepdim=True)
        )

        # Embeddings
        self.static_cat_embedder = (
            FeatureEmbedder(
                cardinalities=static_cardinalities,
                embedding_dims=static_embedding_dim,
            )
            if len(static_cardinalities) > 0
            else None
        )
        self.dynamic_cat_embedder = (
            FeatureEmbedder(
                cardinalities=dynamic_cardinalities,
                embedding_dims=dynamic_embedding_dim,
            )
            if len(dynamic_cardinalities) > 0
            else None
        )
        self.past_dynamic_cat_embedder = (
            FeatureEmbedder(
                cardinalities=past_dynamic_cardinalities,
                embedding_dims=past_dynamic_embedding_dim,
            )
            if len(past_dynamic_cardinalities) > 0
            else None
        )
        self.encoder_in_proj = nn.Linear(
            in_features=self.encoder_dim, out_features=d_model
        )
        self.decoder_in_proj = nn.Linear(
            in_features=self.decoder_dim, out_features=d_model
        )

        # Transformer
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_layers=num_encoder_layers,
            norm_first=True,
        )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_layers=num_decoder_layers,
            norm_first=True,
        )

    @cached_property
    def encoder_dim(self) -> int:
        return (
            self.target_dim
            * (len(self.lags_seq) + 1)  # encoder considers current time step
            + self.time_dim
            + self.static_dim
            + self.dynamic_dim
            + self.past_dynamic_dim
            + sum(self.static_embedding_dim)
            + sum(self.dynamic_embedding_dim)
            + sum(self.past_dynamic_embedding_dim)
            + self.target_dim  # log(scale)
        )

    @cached_property
    def decoder_dim(self) -> int:
        return (
            +self.time_dim
            + self.static_dim
            + self.dynamic_dim
            + sum(self.static_embedding_dim)
            + sum(self.dynamic_embedding_dim)
        )

    @cached_property
    def past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    @staticmethod
    def lagged_sequence_values(
        indices: list[int],
        prior_sequence: Tensor,
        sequence: Tensor,
        dim: int,
    ) -> Tensor:
        lags = lagged_sequence_values(indices, prior_sequence, sequence, dim)
        if lags.dim() > 3:
            lags = lags.reshape(lags.shape[0], lags.shape[1], -1)
        return lags

    def create_encoder_inputs(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        feat_static_real: Optional[Tensor] = None,
        feat_dynamic_real: Optional[Tensor] = None,
        past_feat_dynamic_real: Optional[Tensor] = None,
        feat_static_cat: Optional[Tensor] = None,
        feat_dynamic_cat: Optional[Tensor] = None,
        past_feat_dynamic_cat: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Targets
        context = past_target[:, -self.context_length :]
        observed_context = past_observed_values[:, -self.context_length :]
        scaled_context, loc, scale = self.scaler(context, observed_context)
        scaled_pre_context = (past_target[:, : -self.context_length] - loc) / scale
        encoder_targets = self.lagged_sequence_values(
            [0] + self.lags_seq, scaled_pre_context, scaled_context, dim=1
        )

        # Features
        time_feat = past_time_feat[:, -self.context_length :]
        log_scale = torch.log(scale).view(scale.shape[0], -1)
        static_feats = [log_scale]
        dynamic_feats = [time_feat]
        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if feat_dynamic_real is not None:
            dynamic_feats.append(
                feat_dynamic_real[
                    :, self.past_length - self.context_length : self.past_length
                ]
            )
        if past_feat_dynamic_real is not None:
            dynamic_feats.append(past_feat_dynamic_real[:, -self.context_length :])
        if feat_static_cat is not None and self.static_cat_embedder is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        if feat_dynamic_cat is not None and self.dynamic_cat_embedder is not None:
            dynamic_cat_embed = self.dynamic_cat_embedder(
                feat_dynamic_cat[
                    :, self.past_length - self.context_length : self.past_length
                ]
            )
            dynamic_feats.append(dynamic_cat_embed)
        if (
            past_feat_dynamic_cat is not None
            and self.past_dynamic_cat_embedder is not None
        ):
            past_dynamic_cat_embed = self.past_dynamic_cat_embedder(
                past_feat_dynamic_cat[:, -self.context_length :]
            )
            dynamic_feats.append(past_dynamic_cat_embed)
        static_feats = unsqueeze_expand(
            torch.cat(static_feats, dim=-1), dim=1, size=self.context_length
        )
        dynamic_feats = torch.cat(dynamic_feats, dim=-1)
        encoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)
        return encoder_targets, encoder_feats, loc, scale

    def create_decoder_inputs(
        self,
        future_time_feat: Tensor,
        feat_static_real: Optional[Tensor] = None,
        feat_dynamic_real: Optional[Tensor] = None,
        feat_static_cat: Optional[Tensor] = None,
        feat_dynamic_cat: Optional[Tensor] = None,
    ) -> Tensor:
        # Features
        static_feats = []
        dynamic_feats = [future_time_feat]

        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if feat_dynamic_real is not None:
            dynamic_feats.append(feat_dynamic_real[:, -self.prediction_length :])
        if feat_static_cat is not None and self.static_cat_embedder is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        if feat_dynamic_cat is not None and self.dynamic_cat_embedder is not None:
            dynamic_feats.append(
                self.dynamic_cat_embedder(
                    feat_dynamic_cat[:, -self.prediction_length :]
                )
            )

        dynamic_feats = torch.cat(dynamic_feats, dim=-1)
        if len(static_feats) > 0:
            static_feats = unsqueeze_expand(
                torch.cat(static_feats, dim=-1), dim=1, size=self.prediction_length
            )
            decoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)
        else:
            decoder_feats = dynamic_feats
        return decoder_feats

    def representations(
        self,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
    ) -> dict[str, Tensor]:
        encoder_targets, encoder_feats, loc, scale = self.create_encoder_inputs(
            past_target,
            past_observed_values,
            past_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )
        decoder_feats = self.create_decoder_inputs(
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
        )
        encoder_inputs = self.encoder_in_proj(
            torch.cat([encoder_targets, encoder_feats], dim=-1)
        )
        decoder_inputs = self.decoder_in_proj(decoder_feats)
        memory = self.encoder(
            encoder_inputs,
            attn_mask=attn_mask(
                past_observed_values[:, -self.context_length :],
                is_causal=False,
                device=encoder_inputs.device,
            ),
        )

        representations = self.decoder(
            decoder_inputs,
            memory,
            tgt_is_causal=True,
            memory_mask=attn_mask(
                past_observed_values[:, -self.context_length :],
                is_causal=False,
                query_length=decoder_inputs.size(1),
                device=decoder_inputs.device,
            ),
        )
        return {
            "representations": representations,
            "loc": loc,
            "scale": scale,
        }

    def loss(
        self,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
        loss_fn: DistributionLoss = NegativeLogLikelihood(),
    ) -> dict[str, Tensor]:
        out_dict = self.representations(
            future_target,
            future_observed_values,
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )

        out = out_dict["representations"]
        loc = out_dict["loc"]
        scale = out_dict["scale"]

        distr_params = self.out_proj(out)
        distr = self.distr_output.distribution(distr_params, loc=loc, scale=scale)

        if self.target_shape:
            future_observed_values = future_observed_values.min(dim=-1).values

        loss_per_sample = weighted_average(
            loss_fn(distr, future_target),
            future_observed_values,
            dim=1,
        )
        return loss_per_sample.mean()

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: torch.Tensor,
        future_time_feat: torch.Tensor,
        feat_static_real: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_dynamic_cat: Optional[torch.Tensor] = None,
        past_feat_dynamic_cat: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        num_parallel_samples = num_parallel_samples or self.num_parallel_samples

        encoder_targets, encoder_feats, loc, scale = self.create_encoder_inputs(
            past_target,
            past_observed_values,
            past_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )
        encoder_inputs = self.encoder_in_proj(
            torch.cat([encoder_targets, encoder_feats], dim=-1)
        )

        memory = self.encoder(
            encoder_inputs,
            attn_mask=attn_mask(
                past_observed_values[:, -self.context_length :],
                is_causal=False,
                device=encoder_inputs.device,
            ),
        )

        decoder_feats = self.create_decoder_inputs(
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
        )
        decoder_inputs = self.decoder_in_proj(decoder_feats)

        representations = self.decoder(
            decoder_inputs,
            memory,
            tgt_is_causal=True,
            memory_mask=attn_mask(
                past_observed_values[:, -self.context_length :],
                is_causal=False,
                query_length=decoder_inputs.size(1),
                device=decoder_inputs.device,
            ),
        )

        distr_params = self.out_proj(representations)
        distr = self.distr_output.distribution(distr_params, loc=loc, scale=scale)
        preds = distr.sample(torch.Size((num_parallel_samples,)))
        return rearrange(preds, "n b ... -> b n ...")
