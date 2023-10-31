from functools import cached_property
from typing import Optional

import torch
from einops import rearrange
from gluonts.itertools import prod
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.quantile_output import QuantileOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    unsqueeze_expand,
    weighted_average,
)
from torch import nn, Tensor

from pretraining.model.backbone.layers.transformer import TransformerEncoder
from util.torch.scaler import StdScaler, NOPScaler
from util.torch.attn_mask import attn_mask
from util.torch.ops import unsqueeze_dim, block
from util.torch.distributions import (
    IndependentStudentTOutput,
    MultivariateStudentTOutput,
    SQFOutput,
    ISQFOutput,
    FlowOutput,
)


class MaskedEncoderModel(nn.Module):
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
        distr_output: DistributionOutput | QuantileOutput = StudentTOutput(),
        num_parallel_samples: int = 100,
        quantiles: Optional[list[float]] = None,
        # PEs
        positional_encoding: Optional[str] = None,
        # Attn Mask
        attn_mask_type: Optional[str] = None,
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
        self.past_dynamic_dim = 0
        self.static_cardinalities = static_cardinalities
        self.dynamic_cardinalities = dynamic_cardinalities
        self.past_dynamic_cardinalities = []
        self.static_embedding_dim = static_embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        self.past_dynamic_embedding_dim = []
        self.lags_seq = lags_seq
        self.num_parallel_samples = num_parallel_samples
        self.quantiles = quantiles or (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

        self.scaling = scaling
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout

        # Output
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

        # Transformer
        use_sinusoidal_embeds = False
        use_learned_embeds = False
        use_rotary_embeds = False
        use_scaled_rotary_embeds = False
        max_len = None
        interp_len = None
        if positional_encoding is None:
            pass
        elif positional_encoding == "sinusoidal":
            use_sinusoidal_embeds = True
            max_len = context_length + prediction_length
        elif positional_encoding == "learned":
            use_learned_embeds = True
            max_len = context_length + prediction_length
        elif positional_encoding == "sinusoidal_interpolation":
            use_sinusoidal_embeds = True
            max_len = context_length + prediction_length
            interp_len = 480 + 48  # hardcoded to experiments
        elif positional_encoding == "rotary":
            use_rotary_embeds = True
        elif positional_encoding == "scaled_rotary":
            use_scaled_rotary_embeds = True
        else:
            raise ValueError(
                f"positional_encoding must be one of [sinusoidal, sinusoidal_interpolation, alibi, rotary, scaled_rotary], "
                f"got {positional_encoding}"
            )

        self.decoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_layers=num_encoder_layers,
            norm_first=True,
            max_len=max_len,
            interp_len=interp_len,
            use_sinusoidal_embeds=use_sinusoidal_embeds,
            use_learned_embeds=use_learned_embeds,
            use_rotary_embeds=use_rotary_embeds,
            use_scaled_rotary_embeds=use_scaled_rotary_embeds,
        )
        self.attn_mask_type = attn_mask_type

        # Embeddings
        self.mask = nn.Embedding(1, d_model)
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

        self.decoder_in_proj = nn.Linear(
            in_features=self.decoder_dim, out_features=d_model
        )

    @cached_property
    def decoder_dim(self) -> int:
        return (
            self.target_dim
            * (len(self.lags_seq) + 1)  # encoder considers current time step
            + self.time_dim
            + self.static_dim
            + self.dynamic_dim
            + sum(self.static_embedding_dim)
            + sum(self.dynamic_embedding_dim)
            + self.target_dim  # log(scale)
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

    @property
    def mask_token(self) -> Tensor:
        return self.mask.weight.unsqueeze(0)

    def get_attn_mask(self, past_observed_values: Tensor, future_observed_values: Tensor) -> Tensor:
        if self.attn_mask_type is None:
            mask = attn_mask(
                torch.cat(
                    [
                        past_observed_values[:, -self.context_length:],
                        future_observed_values,
                    ],
                    dim=1,
                ),
                device=past_observed_values.device,
            )
        elif self.attn_mask_type == "full_causal":
            mask = attn_mask(
                torch.cat(
                    [
                        torch.ones_like(past_observed_values[:, -self.context_length:]),
                        future_observed_values,
                    ],
                    dim=1,
                ),
                is_causal=True,
                device=past_observed_values.device,
            )
        elif self.attn_mask_type == "decoder_causal":
            context_prediction_query_context_key = attn_mask(
                past_observed_values[:, -self.context_length:],
                query_length=self.context_length + future_observed_values.size(1),
                device=past_observed_values.device,
            )
            context_query_prediction_key = block(
                True,
                self.context_length,
                sz2=future_observed_values.size(1),
                bsz=(past_observed_values.size(0),),
                device=past_observed_values.device,
            )
            prediction_query_prediction_key = attn_mask(
                future_observed_values, is_causal=True, device=past_observed_values.device
            )
            context_prediction_query_prediction_key = torch.cat(
                [context_query_prediction_key, prediction_query_prediction_key], dim=1
            )
            mask = torch.cat([context_prediction_query_context_key, context_prediction_query_prediction_key], dim=-1)
        else:
            raise ValueError(
                f"attn_mask_type must be one of [None, full_causal, decoder_causal], got {self.attn_mask_type}"
            )
        return mask

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
        log_scale = torch.log(scale).view(scale.shape[0], -1)
        static_feats = [log_scale]
        if self.time_dim > 0:
            time_feat = past_time_feat[:, -self.context_length:]
            dynamic_feats = [time_feat]
        else:
            dynamic_feats = []
        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if feat_dynamic_real is not None:
            dynamic_feats.append(
                feat_dynamic_real[
                    :, self.past_length - self.context_length : self.past_length
                ]
            )
        if feat_static_cat is not None and self.static_cat_embedder is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        if feat_dynamic_cat is not None and self.dynamic_cat_embedder is not None:
            dynamic_cat_embed = self.dynamic_cat_embedder(
                feat_dynamic_cat[
                    :, self.past_length - self.context_length : self.past_length
                ]
            )
            dynamic_feats.append(dynamic_cat_embed)
        static_feats = unsqueeze_expand(
            torch.cat(static_feats, dim=-1), dim=1, size=self.context_length
        )
        if len(dynamic_feats) > 0:
            dynamic_feats = torch.cat(dynamic_feats, dim=-1)
            encoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)
        else:
            encoder_feats = static_feats
        return encoder_targets, encoder_feats, loc, scale

    def create_decoder_inputs(
        self,
        scale: Tensor,
        future_time_feat: Tensor,
        feat_static_real: Optional[Tensor] = None,
        feat_dynamic_real: Optional[Tensor] = None,
        feat_static_cat: Optional[Tensor] = None,
        feat_dynamic_cat: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        # Features
        log_scale = torch.log(scale).view(scale.shape[0], -1)
        static_feats = [log_scale]

        if self.time_dim > 0:
            dynamic_feats = [future_time_feat]
        else:
            dynamic_feats = []

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
        static_feats = unsqueeze_expand(
            torch.cat(static_feats, dim=-1), dim=1, size=self.prediction_length
        )

        if len(dynamic_feats) > 0:
            dynamic_feats = torch.cat(dynamic_feats, dim=-1)
            decoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)
        else:
            decoder_feats = static_feats

        target_dim = self.decoder_dim - decoder_feats.size(-1)
        decoder_targets = torch.zeros(
            (decoder_feats.size(0), self.prediction_length, target_dim),
            device=decoder_feats.device,
        )

        return decoder_targets, decoder_feats

    def representations(
        self,
        future_target: Tensor,
        future_observed_values: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        feat_static_real: Optional[Tensor] = None,
        feat_dynamic_real: Optional[Tensor] = None,
        past_feat_dynamic_real: Optional[Tensor] = None,
        feat_static_cat: Optional[Tensor] = None,
        feat_dynamic_cat: Optional[Tensor] = None,
        past_feat_dynamic_cat: Optional[Tensor] = None,
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
        decoder_targets, decoder_feats = self.create_decoder_inputs(
            scale,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
        )
        encoder_inputs = self.decoder_in_proj(
            torch.cat([encoder_targets, encoder_feats], dim=-1)
        )
        decoder_inputs = (
            self.decoder_in_proj(torch.cat([decoder_targets, decoder_feats], dim=-1))
            + self.mask_token
        )
        representations = self.decoder(
            torch.cat([encoder_inputs, decoder_inputs], dim=1),
            attn_mask=self.get_attn_mask(past_observed_values, future_observed_values),
        )[:, -self.prediction_length :]
        return {
            "representations": representations,
            "loc": loc,
            "scale": scale,
        }

    def loss(
        self,
        future_target: Tensor,
        future_observed_values: Tensor,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        feat_static_real: Optional[Tensor] = None,
        feat_dynamic_real: Optional[Tensor] = None,
        past_feat_dynamic_real: Optional[Tensor] = None,
        feat_static_cat: Optional[Tensor] = None,
        feat_dynamic_cat: Optional[Tensor] = None,
        past_feat_dynamic_cat: Optional[Tensor] = None,
        loss_fn: DistributionLoss = NegativeLogLikelihood(),
    ) -> Tensor:

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

        if isinstance(self.distr_output, DistributionOutput):
            distr_params = self.out_proj(out)
            preds = self.distr_output.distribution(distr_params, loc=loc, scale=scale)
            loss_per_dim = loss_fn(preds, future_target)
        elif isinstance(self.distr_output, QuantileOutput):
            preds = self.out_proj(out) * scale + loc
            loss_per_dim = self.distr_output.quantile_loss(
                preds, future_target
            )
        else:
            raise ValueError(
                f"Unknown distr_output type {type(self.distr_output).__name__}."
            )

        if self.target_shape:
            future_observed_values = future_observed_values.min(dim=-1).values

        if len(loss_per_dim.shape) > len(future_observed_values.shape):
            if isinstance(self.distr_output, (QuantileOutput, SQFOutput, ISQFOutput)):
                loss_per_dim = loss_per_dim.mean(-1)
            else:
                loss_per_dim = loss_per_dim.sum(-1)

        loss_per_batch = weighted_average(
            loss_per_dim,
            future_observed_values,
            dim=1,
        )
        return loss_per_batch.mean()

    def forward(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        feat_static_real: Optional[Tensor] = None,
        feat_dynamic_real: Optional[Tensor] = None,
        past_feat_dynamic_real: Optional[Tensor] = None,
        feat_static_cat: Optional[Tensor] = None,
        feat_dynamic_cat: Optional[Tensor] = None,
        past_feat_dynamic_cat: Optional[Tensor] = None,
        num_parallel_samples: Optional[int] = None,
        quantiles: Optional[list[float]] = None,
    ) -> Tensor:
        num_parallel_samples = num_parallel_samples or self.num_parallel_samples
        quantiles = quantiles or self.quantiles

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
        encoder_inputs = self.decoder_in_proj(
            torch.cat([encoder_targets, encoder_feats], dim=-1)
        )
        decoder_targets, decoder_feats = self.create_decoder_inputs(
            scale,
            future_time_feat,
            feat_static_real,
            feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
        )

        future_observed_values = torch.ones(
            (past_observed_values.size(0), self.prediction_length)
            + past_observed_values.shape[2:],
            device=past_observed_values.device,
        )

        decoder_inputs = (
            self.decoder_in_proj(torch.cat([decoder_targets, decoder_feats], dim=-1))
            + self.mask_token
        )
        representations = self.decoder(
            torch.cat([encoder_inputs, decoder_inputs], dim=1),
            attn_mask=self.get_attn_mask(past_observed_values, future_observed_values),
        )[:, -self.prediction_length :]

        if isinstance(self.distr_output, QuantileOutput):
            preds = self.out_proj(representations) * scale + loc
        else:
            distr_params = self.out_proj(representations)
            if isinstance(
                self.distr_output,
                (StudentTOutput, MultivariateStudentTOutput, IndependentStudentTOutput, FlowOutput),
            ):
                distr = self.distr_output.distribution(distr_params, loc=loc, scale=scale)
                preds = distr.sample(torch.Size((num_parallel_samples,)))
            elif isinstance(self.distr_output, (SQFOutput, ISQFOutput)):
                distr = self.distr_output.distribution(distr_params)
                quantiles = unsqueeze_dim(
                    torch.as_tensor(
                        quantiles, dtype=past_target.dtype, device=past_target.device
                    ),
                    future_time_feat.shape[:-1] + self.distr_output.event_shape,
                )
                preds = distr.quantile(quantiles)
                preds = loc + scale * preds
            else:
                raise NotImplementedError(
                    f"Unknown distr_output type: {self.distr_output.__class__.__name__}"
                )

        return rearrange(preds, "n b ... -> b n ...")
