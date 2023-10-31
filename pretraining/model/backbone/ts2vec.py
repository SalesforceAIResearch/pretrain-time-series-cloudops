from functools import cached_property
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from gluonts.itertools import prod
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.distributions.distribution_output import PtArgProj
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.lambda_layer import LambdaLayer
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import (
    lagged_sequence_values,
    unsqueeze_expand,
    weighted_average,
)

from util.torch.scaler import StdScaler, NOPScaler


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, extract_layers=None):
        super().__init__()

        if extract_layers is not None:
            assert len(channels) - 1 in extract_layers

        self.extract_layers = extract_layers
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        if self.extract_layers is not None:
            outputs = []
            for idx, mod in enumerate(self.net):
                x = mod(x)
                if idx in self.extract_layers:
                    outputs.append(x)
            return outputs
        return self.net(x)


class TS2VecModel(nn.Module):
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
            mode: str = "pretrain",
            hidden_dims: int = 64,
            output_dims: int = 320,
            depth: int = 10,
            temporal_unit: int = 0,
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
        self.past_dynamic_cardinalities = []
        self.static_embedding_dim = static_embedding_dim
        self.dynamic_embedding_dim = dynamic_embedding_dim
        self.past_dynamic_embedding_dim = []
        self.lags_seq = lags_seq
        self.num_parallel_samples = num_parallel_samples

        self.mode = mode
        self.temporal_unit = temporal_unit

        self.encoder = nn.Sequential(
            Rearrange('b t d -> b d t'),
            DilatedConvEncoder(
                hidden_dims,
                [hidden_dims] * depth + [output_dims],
                kernel_size=3
            ),
            Rearrange('b d t -> b t d'),
            nn.Dropout(0.1),
        )

        # Output distribution
        self.target_shape = distr_output.event_shape
        self.target_dim = prod(self.target_shape)
        self.distr_output = distr_output
        self.out_proj = nn.Sequential(
            LambdaLayer(lambda x: x[:, -1]),
            PtArgProj(
                in_features=output_dims,
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
        ) if mode != "pretrain" else None
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
            in_features=self.encoder_dim, out_features=hidden_dims
        )

        if mode != "pretrain":
            for param in self.parameters():
                param.requires_grad = False
            for param in self.out_proj.parameters():
                param.requires_grad = True

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

    def pretrain_loss(
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
        loss_fn = None,
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

        ts_l = encoder_inputs.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=encoder_inputs.size(0))

        inp1 = take_per_row(encoder_inputs, crop_offset + crop_eleft, crop_right - crop_eleft)
        inp2 = take_per_row(encoder_inputs, crop_offset + crop_left, crop_eright - crop_left)

        inp1[~generate_binomial_mask(inp1.size(0), inp1.size(1))] = 0
        inp2[~generate_binomial_mask(inp2.size(0), inp2.size(1))] = 0

        out1 = self.encoder(inp1)[:, -crop_l:]
        out2 = self.encoder(inp2)[:, :crop_l]

        loss = hierarchical_contrastive_loss(
            out1,
            out2,
            temporal_unit=self.temporal_unit
        )
        return loss

    def train_loss(
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
        representations = self.encoder(encoder_inputs)
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

    def loss(self, *args, **kwargs) -> Tensor:
        if self.mode == "finetune":
            return self.train_loss(*args, **kwargs)
        else:
            return self.pretrain_loss(*args, **kwargs)

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
        representations = self.encoder(encoder_inputs)
        if self.out_proj is None:
            return representations
        distr_params = self.out_proj(representations)
        distr = self.distr_output.distribution(distr_params, loc=loc, scale=scale)
        preds = distr.sample(torch.Size((num_parallel_samples,)))
        return rearrange(preds, "n b ... -> b n ...")
