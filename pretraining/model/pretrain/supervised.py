from pathlib import Path
from typing import Optional, Type, Any

import torch
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from torch import nn
from transformers import get_cosine_schedule_with_warmup

from util.torch.lightning_module import LightningModule
from pretraining.model.backbone.cost import BandedFourierLayer


class Supervised(LightningModule):
    @validated()
    def __init__(
        self,
        # Model arguments
        backbone_cls: Type[nn.Module],
        backbone_args: dict[str, Any],
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        static_dim: int = 0,
        dynamic_dim: int = 0,
        past_dynamic_dim: int = 0,
        static_cardinalities: Optional[list[int]] = None,
        dynamic_cardinalities: Optional[list[int]] = None,
        past_dynamic_cardinalities: Optional[list[int]] = None,
        static_embedding_dim: Optional[list[int]] = None,
        dynamic_embedding_dim: Optional[list[int]] = None,
        past_dynamic_embedding_dim: Optional[list[int]] = None,
        time_features: Optional[list[TimeFeature] | str] = None,
        lags_seq: Optional[list[int]] = None,
        scaling: bool = True,
        # Supervised arguments
        loss: DistributionLoss = NegativeLogLikelihood(),
        # Optimizer
        lr: float = 1e-3,
        lr_warmup: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.98,
        weight_decay: float = 1e-2,
        num_train_steps: int = 10000,
        mask_range: Optional[int] = None,
    ):
        super().__init__(
            freq,
            prediction_length,
            backbone_args["distr_output"].event_shape,
            context_length=context_length,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            past_dynamic_dim=past_dynamic_dim,
            static_cardinalities=static_cardinalities,
            dynamic_cardinalities=dynamic_cardinalities,
            past_dynamic_cardinalities=past_dynamic_cardinalities,
            static_embedding_dim=static_embedding_dim,
            dynamic_embedding_dim=dynamic_embedding_dim,
            past_dynamic_embedding_dim=past_dynamic_embedding_dim,
            time_features=time_features,
            lags_seq=lags_seq,
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.save_hyperparameters()
        torch.distributions.Distribution.set_default_validate_args(False)
        self.backbone = backbone_cls(
            self.freq,
            self.context_length,
            self.prediction_length,
            self.time_dim,
            self.static_dim,
            self.dynamic_dim,
            self.past_dynamic_dim,
            self.static_cardinalities,
            self.dynamic_cardinalities,
            self.past_dynamic_cardinalities,
            self.static_embedding_dim,
            self.dynamic_embedding_dim,
            self.past_dynamic_embedding_dim,
            self.lags_seq,
            self.scaling,
            **backbone_args,
        )
        self.loss_fn = loss
        self.mask_range = mask_range

    def from_pretrained(self, ckpt: str | Path, prefix: str = "backbone."):
        state_dict = torch.load(ckpt)["state_dict"]
        backbone_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                backbone_state_dict[k[len(prefix) :]] = v
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            backbone_state_dict, strict=False
        )
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.backbone.loss(*args, **kwargs, loss_fn=self.loss_fn)

    def configure_optimizers(self) -> any:
        """
        Returns the optimizer to use.
        """
        decay = set()
        no_decay = set()

        whitelist = (nn.Linear, nn.Conv1d, BandedFourierLayer)
        blacklist = (nn.BatchNorm1d, nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    # weights of whitelist will be decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    # weights of blacklist will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("log_gamma") or pn.endswith("beta") or pn.endswith("_lambda"):
                    # normalizing flow stuff
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        warmup_steps = (
            int(self.hparams.lr_warmup * self.hparams.num_train_steps)
            if self.hparams.lr_warmup < 1.0
            else self.hparams.lr_warmup
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            self.hparams.num_train_steps,
            num_cycles=0.5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }
