from typing import Callable, Optional, Any, Type

import torch
from gluonts.core.component import validated
from gluonts.time_feature import get_seasonality
from torch import Tensor, nn

from util.torch.lightning_module import LightningModule


class MetaNBEATS(LightningModule):
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

        self.model = backbone_cls(
            self.prediction_length,
            self.context_length,
            scale=scaling,
            **backbone_args
        )
        self.periodicity = get_seasonality(freq)
        self.loss_fn = self.get_loss_fn(loss)

    def mape_loss(
        self,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        denominator = torch.abs(future_target)
        flag = (denominator == 0).float()

        absolute_error = torch.abs(future_target - forecast) * future_observed_values

        mape = (100 / self.prediction_length) * torch.mean(
            (absolute_error * (1 - flag)) / (denominator + flag),
            dim=1,
        )
        return mape

    def smape_loss(
        self,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        # Stop gradient required for stable learning
        denominator = (torch.abs(future_target) + torch.abs(forecast)).detach()
        flag = (denominator == 0).float()

        absolute_error = torch.abs(future_target - forecast) * future_observed_values

        smape = (200 / self.prediction_length) * torch.mean(
            (absolute_error * (1 - flag)) / (denominator + flag),
            dim=1,
        )

        return smape

    def mase_loss(
        self,
        forecast: Tensor,
        future_target: Tensor,
        past_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        whole_target = torch.cat([past_target, future_target], dim=1)
        seasonal_error = torch.mean(
            torch.abs(
                whole_target[:, self.periodicity :]
                - whole_target[:, : -self.periodicity]
            ),
            dim=1,
        )
        flag = (seasonal_error == 0).float()

        absolute_error = torch.abs(future_target - forecast) * future_observed_values

        mase = (torch.mean(absolute_error, dim=1) * (1 - flag)) / (
            seasonal_error + flag
        )

        return mase

    def get_loss_fn(
        self, loss: str
    ) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        if loss == "smape":
            return self.smape_loss
        elif loss == "mape":
            return self.mape_loss
        elif loss == "mase":
            return self.mase_loss
        else:
            raise ValueError(
                f"Unknown loss function: {loss}, "
                f"loss function should be one of ('mse', 'mae')."
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def loss(self, *args, **kwargs) -> torch.Tensor:
        forecast = self.model(kwargs["past_target"], kwargs["past_observed_values"])
        return self.loss_fn(
            forecast,
            kwargs["future_target"],
            kwargs["past_target"],
            kwargs["future_observed_values"],
        ).mean()
