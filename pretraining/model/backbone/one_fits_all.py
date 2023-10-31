from typing import Optional
from functools import cached_property

import torch
from einops import rearrange
from torch import nn, Tensor
from gluonts.itertools import prod
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.distributions.distribution_output import PtArgProj
from gluonts.torch.modules.lambda_layer import LambdaLayer
from gluonts.torch.util import unsqueeze_expand, weighted_average
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from util.torch.scaler import StdScaler, NOPScaler


class OneFitsAllModel(nn.Module):
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
        # Patching
        patch_length: int = 16,
        stride: int = 8,
        num_parallel_samples: int = 100,
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

        num_patches = (context_length - patch_length) / stride + 1
        assert num_patches % 1 == 0, "num_patches must be an integer. Please adjust patch_length and stride."
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = int(num_patches)

        gpt2_config = GPT2Config.from_pretrained("gpt2")

        # Output distribution
        self.target_shape = distr_output.event_shape
        self.target_dim = prod(self.target_shape)
        self.distr_output = distr_output

        self.out_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            PtArgProj(
                in_features=gpt2_config.n_embd * self.num_patches,
                args_dim={
                    k: prediction_length * v for k, v in distr_output.args_dim.items()
                },
                domain_map=LambdaLayer(
                    (
                        (
                            lambda *args: distr_output.domain_map(
                                *[
                                    rearrange(
                                        arg,
                                        "... (n d) -> ... n d",
                                        n=prediction_length,
                                    )
                                    for arg in args
                                ]
                            )
                        )
                        if len(self.target_shape) > 0
                        else distr_output.domain_map
                    )
                ),
            ),
        )

        self.encoder_in_proj = nn.Linear(
            in_features=self.encoder_dim, out_features=gpt2_config.n_embd
        )
        self.gpt2 = GPT2Model.from_pretrained("gpt2")

        # only fine-tune layer norm and positional embeddings
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

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

    @cached_property
    def encoder_dim(self) -> int:
        return (
            self.target_dim
            * self.patch_length
            + self.static_dim
            + self.past_dynamic_dim
            * self.patch_length
            + self.target_dim  # log(scale)
        )

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
        context = past_target[:, -self.context_length:]
        observed_context = past_observed_values[:, -self.context_length:]
        scaled_context, loc, scale = self.scaler(context, observed_context)

        scaled_context = scaled_context.unfold(
            dimension=1, size=self.patch_length, step=self.stride
        ).flatten(start_dim=2)

        # Features
        log_scale = torch.log(scale).view(scale.shape[0], -1)
        static_feats = [log_scale]
        dynamic_feats = []
        if feat_static_real is not None:
            static_feats.append(feat_static_real)
        if past_feat_dynamic_real is not None:
            dynamic_feats.append(
                past_feat_dynamic_real[:, -self.context_length:].unfold(
                    dimension=1, size=self.patch_length, step=self.stride
                ).flatten(start_dim=2)
            )
        if feat_static_cat is not None and self.static_cat_embedder is not None:
            static_feats.append(self.static_cat_embedder(feat_static_cat))
        static_feats = unsqueeze_expand(
            torch.cat(static_feats, dim=-1), dim=1, size=self.num_patches
        )
        if len(dynamic_feats) > 0:
            dynamic_feats = torch.cat(dynamic_feats, dim=-1)
            encoder_feats = torch.cat([static_feats, dynamic_feats], dim=-1)
        else:
            encoder_feats = static_feats
        return scaled_context, encoder_feats, loc, scale

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
        representations = self.gpt2(
            inputs_embeds=encoder_inputs, use_cache=False
        ).last_hidden_state

        distr_params = self.out_proj(representations)
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
    ) -> Tensor:
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

        representations = self.gpt2(
            inputs_embeds=encoder_inputs, use_cache=False
        ).last_hidden_state
        distr_params = self.out_proj(representations)
        distr = self.distr_output.distribution(distr_params, loc=loc, scale=scale)
        preds = distr.sample(torch.Size((num_parallel_samples,)))
        return rearrange(preds, "n b ... -> b n ...")
