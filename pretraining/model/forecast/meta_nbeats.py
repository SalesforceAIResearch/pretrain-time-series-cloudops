import copy
from typing import Callable, Optional, Any, Type
from collections import defaultdict

import torch
from torch import nn
from gluonts.core.component import validated

from util.torch.lightning_module import LightningModule
from pretraining.model.pretrain.meta_nbeats import MetaNBEATS


class NBEATSEnsemble(LightningModule):
    @validated()
    def __init__(
        self,
        backbone_cls: Type[nn.Module],
        backbone_args: dict[str, Any],
        freq: str,
        prediction_length: int,
        static_dim: int = 0,
        past_dynamic_dim: int = 0,
        static_cardinalities: Optional[list[int]] = None,
        scaling: bool = True,
        context_len_mult: int = 9,
        checkpoints: list[str] = [],
        # Training
        loss: str = "smape",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
        num_train_steps: int = 10000,
    ):
        context_length = prediction_length * context_len_mult
        super().__init__(
            freq,
            prediction_length,
            (),
            context_length=context_length,
            static_dim=0,
            dynamic_dim=0,
            past_dynamic_dim=0,
            static_cardinalities=None,
            dynamic_cardinalities=None,
            past_dynamic_cardinalities=None,
            static_embedding_dim=None,
            dynamic_embedding_dim=None,
            past_dynamic_embedding_dim=None,
            time_features=[],
            age_feature=False,
            lags_seq=[],
            scaling=scaling,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        self.save_hyperparameters()

        models = defaultdict(list)
        for path in checkpoints:
            module = MetaNBEATS.load_from_checkpoint(path)
            group = f"{module.hparams.backbone_args['model_type']},{module.context_length}"
            models[group].append(module.model)
        self.models = torch.nn.ModuleDict(
            {k: torch.nn.ModuleList(v) for k, v in models.items()}
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        forecasts = []
        for group, models in self.models.items():
            context_length = int(group.split(",")[-1])
            base_model = copy.deepcopy(models[0])
            base_model.to("meta")
            params, buffers = torch.func.stack_module_state(models)

            def call_model(p, b, past_target, past_observed_values):
                return torch.func.functional_call(
                    base_model,
                    (p, b),
                    (past_target, past_observed_values),
                )

            past_target = kwargs["past_target"][:, -context_length:]
            past_observed_values = kwargs["past_observed_values"][:, -context_length:]
            forecast = torch.vmap(call_model, (0, 0, None, None))(
                params, buffers, past_target, past_observed_values
            )
            forecasts.append(forecast)

        return torch.cat(forecasts, dim=0).transpose(0, 1)
