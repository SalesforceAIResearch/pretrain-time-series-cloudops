import math
from functools import cached_property
from typing import Optional

import torch
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from torch import nn, Tensor
from einops import rearrange, reduce
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


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        weight = torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat)
        bias = torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
        self.weight = nn.Parameter(torch.view_as_real(weight))
        self.bias = nn.Parameter(torch.view_as_real(bias))
        # self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], torch.view_as_complex(self.weight))
        return output + torch.view_as_complex(self.bias)

    # def reset_parameters(self) -> None:
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #     nn.init.uniform_(self.bias, -bound, bound)


class CoSTEncoder(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        kernels: list[int],
        length: int,
        hidden_dims=64,
        depth=10,
    ):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k - 1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False):  # x: B x T x input_dims
        x = self.input_fc(x)  # B x T x Ch

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)


class CoSTModel(nn.Module):
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
            kernels: list[int] = [1, 2, 4, 8, 16, 32, 64, 128],
            alpha: Optional[float] = 0.0005,
            K: Optional[int] = 256,
            m: Optional[float] = 0.999,
            T: Optional[float] = 0.07,
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
        self.alpha = alpha
        self.m = m
        self.T = T
        self.K = K

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

        self.encoder_q = CoSTEncoder(
            self.encoder_dim,
            output_dims,
            kernels=kernels,
            length=context_length,
            hidden_dims=hidden_dims,
            depth=depth,
        )
        self.encoder_k = CoSTEncoder(
            self.encoder_dim,
            output_dims,
            kernels=kernels,
            length=context_length,
            hidden_dims=hidden_dims,
            depth=depth,
        )

        # create the encoders
        self.head_q = nn.Sequential(
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims),
            nn.ReLU(),
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims)
        )
        self.head_k = nn.Sequential(
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims),
            nn.ReLU(),
            nn.Linear(self.encoder_q.component_dims, self.encoder_q.component_dims)
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(self.encoder_q.component_dims, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

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

    def transform(self, x, p=0.5, sigma=0.5):
        def jitter(x):
            add = torch.randn(x.shape, device=x.device) * sigma
            mask = torch.rand(x.size(0), device=x.device) < p
            dim = (...,) + (None,) * len(add.shape[1:])
            add = add * mask[dim]
            return x + add

        def scale(x):
            if len(self.target_shape) > 0:
                mult = torch.randn(x.size(0), x.size(-1), device=x.device) * sigma
            else:
                mult = torch.randn(x.size(0), device=x.device) * sigma
            mask = torch.rand(x.size(0), device=x.device) < p
            dim = (...,) + (None,) * len(mult.shape[1:])
            mult = mult * mask[dim]
            mult = mult[:, None]
            return x * (1 + mult)

        def shift(x):
            if len(self.target_shape) > 0:
                add = torch.randn(x.size(0), x.size(-1), device=x.device) * sigma
            else:
                add = torch.randn(x.size(0), device=x.device) * sigma
            mask = torch.rand(x.size(0), device=x.device) < p
            dim = (...,) + (None,) * len(add.shape[1:])
            add = add * mask[dim]
            add = add[:, None]
            return x + add

        return jitter(shift(scale(x)))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update for key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def compute_loss(self, q, k, k_negs):
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators - first dim of each batch
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = F.cross_entropy(logits, labels)

        return loss

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

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
        loss_fn=None,
    ) -> Tensor:
        past_target_q = self.transform(past_target)
        past_target_k = self.transform(past_target)
        encoder_targets_q, encoder_feats_q, loc, scale = self.create_encoder_inputs(
            past_target_q,
            past_observed_values,
            past_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )
        encoder_targets_k, encoder_feats_k, loc, scale = self.create_encoder_inputs(
            past_target_k,
            past_observed_values,
            past_time_feat,
            feat_static_real,
            feat_dynamic_real,
            past_feat_dynamic_real,
            feat_static_cat,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
        )

        x_q = torch.cat([encoder_targets_q, encoder_feats_q], dim=-1)
        x_k = torch.cat([encoder_targets_k, encoder_feats_k], dim=-1)
        # compute query features
        rand_idx = np.random.randint(0, x_q.shape[1])

        q_t, q_s = self.encoder_q(x_q)
        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)

        loss = 0

        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(x_k)
        k_s = F.normalize(k_s, dim=-1)

        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        loss += (self.alpha * (seasonal_loss / 2))

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
        encoder_inputs = torch.cat([encoder_targets, encoder_feats], dim=-1)
        trend, season = self.encoder_q(encoder_inputs)
        representations = torch.cat([trend, season], dim=-1)
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
        encoder_inputs = torch.cat([encoder_targets, encoder_feats], dim=-1)
        trend, season = self.encoder_q(encoder_inputs)
        representations = torch.cat([trend, season], dim=-1)
        if self.out_proj is None:
            return representations
        distr_params = self.out_proj(representations)
        distr = self.distr_output.distribution(distr_params, loc=loc, scale=scale)
        preds = distr.sample(torch.Size((num_parallel_samples,)))
        return rearrange(preds, "n b ... -> b n ...")
