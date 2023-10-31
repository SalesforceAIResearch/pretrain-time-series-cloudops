import numpy as np
import torch
from torch import nn, Tensor
from gluonts.core.component import validated

from util.torch.scaler import StdScaler, NOPScaler


class NBEATSBlock(nn.Module):
    @validated()
    def __init__(
        self,
        width: int,
        num_block_layers: int,
        theta_size: int,
        prediction_length: int,
        context_length: int,
        basis_function: nn.Module,
    ):
        super().__init__()

        self.width = width
        self.num_block_layers = num_block_layers
        self.expansion_coefficient_length = theta_size
        self.prediction_length = prediction_length
        self.context_length = context_length

        self.fc_stack = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.context_length, self.width)
                    if i == 0
                    else nn.Linear(self.width, self.width),
                    nn.ReLU(),
                )
                for i in range(self.num_block_layers)
            ]
        )

        self.basis_parameters = nn.Linear(self.width, self.expansion_coefficient_length)
        self.basis_function = basis_function

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = self.fc_stack(x)
        theta = self.basis_parameters(x)
        backcast, forecast = self.basis_function(theta)
        return backcast, forecast


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class SeasonalityBasis(nn.Module):
    @validated()
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()

        frequency = torch.cat(
            [
                torch.zeros(1),
                torch.arange(harmonics, harmonics / 2 * forecast_size) / harmonics,
            ]
        ).unsqueeze(1)

        backcast_grid = (
            -2
            * torch.pi
            * (torch.arange(backcast_size).unsqueeze(0) / backcast_size)
            * frequency
        )

        forecast_grid = (
            2
            * torch.pi
            * (torch.arange(forecast_size).unsqueeze(0) / forecast_size)
            * frequency
        )

        self.register_buffer(
            "backcast_cos_template",
            torch.cos(backcast_grid),
        )
        self.register_buffer(
            "backcast_sin_template",
            torch.sin(backcast_grid),
        )
        self.register_buffer(
            "forecast_cos_template",
            torch.cos(forecast_grid),
        )
        self.register_buffer(
            "forecast_sin_template",
            torch.sin(forecast_grid),
        )

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        params_per_harmonic = theta.shape[1] // 4

        backcast_harmonics_cos = (
            theta[:, :params_per_harmonic] @ self.backcast_cos_template
        )
        backcast_harmonics_sin = (
            theta[:, params_per_harmonic : 2 * params_per_harmonic]
            @ self.backcast_sin_template
        )
        backcast = backcast_harmonics_cos + backcast_harmonics_sin
        forecast_harmonics_cos = (
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic]
            @ self.forecast_cos_template
        )
        forecast_harmonics_sin = (
            theta[:, 3 * params_per_harmonic :] @ self.forecast_sin_template
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos
        return backcast, forecast


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        self.polynomial_size = (
            degree_of_polynomial + 1
        )  # degree of polynomial with constant term
        self.register_buffer(
            "backcast_time",
            torch.cat(
                [
                    torch.pow(torch.arange(backcast_size) / backcast_size, i).unsqueeze(
                        0
                    )
                    for i in range(self.polynomial_size)
                ],
                dim=0,
            ),
        )
        self.register_buffer(
            "forecast_time",
            torch.cat(
                [
                    torch.pow(torch.arange(forecast_size) / forecast_size, i).unsqueeze(
                        0
                    )
                    for i in range(self.polynomial_size)
                ],
                dim=0,
            ),
        )

    def forward(self, theta: Tensor):
        backcast = theta[:, self.polynomial_size :] @ self.backcast_time
        forecast = theta[:, : self.polynomial_size] @ self.forecast_time
        return backcast, forecast


class NBEATSInterpretableModel(nn.Module):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        trend_blocks: int,
        trend_layers: int,
        trend_layer_size: int,
        degree_of_polynomial: int,
        seasonality_blocks: int,
        seasonality_layers: int,
        seasonality_layer_size: int,
        num_of_harmonics: int,
        scale: bool = False,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length

        if scale:
            self.scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        trend_block = NBEATSBlock(
            width=trend_layer_size,
            num_block_layers=trend_layers,
            theta_size=2 * (degree_of_polynomial + 1),
            prediction_length=prediction_length,
            context_length=context_length,
            basis_function=TrendBasis(
                degree_of_polynomial=degree_of_polynomial,
                backcast_size=context_length,
                forecast_size=prediction_length,
            ),
        )
        seasonality_block = NBEATSBlock(
            width=seasonality_layer_size,
            num_block_layers=seasonality_layers,
            theta_size=4
            * int(
                np.ceil(num_of_harmonics / 2 * prediction_length)
                - (num_of_harmonics - 1)
            ),
            prediction_length=prediction_length,
            context_length=context_length,
            basis_function=SeasonalityBasis(
                harmonics=num_of_harmonics,
                backcast_size=context_length,
                forecast_size=prediction_length,
            ),
        )
        self.blocks = nn.ModuleList(
            [trend_block for _ in range(trend_blocks)]
            + [seasonality_block for _ in range(seasonality_blocks)]
        )

    def forward(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
    ) -> Tensor:
        past_target, loc, scale = self.scaler(past_target, past_observed_values)
        residuals = past_target.flip(dims=(1,))
        input_mask = past_observed_values.flip(dims=(1,))
        forecast = past_target[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return loc + forecast * scale


class NBEATSGenericModel(nn.Module):
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        stacks: int,
        layers: int,
        layer_size: int,
        scale: bool = False,
    ):
        super().__init__()

        self.prediction_length = prediction_length
        self.context_length = context_length

        if scale:
            self.scaler = StdScaler(dim=1, keepdim=True)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)

        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    width=layer_size,
                    num_block_layers=layers,
                    theta_size=context_length + prediction_length,
                    prediction_length=prediction_length,
                    context_length=context_length,
                    basis_function=GenericBasis(
                        backcast_size=context_length,
                        forecast_size=prediction_length,
                    ),
                )
                for _ in range(stacks)
            ]
        )

    def forward(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
    ) -> Tensor:
        past_target, loc, scale = self.scaler(past_target, past_observed_values)
        residuals = past_target.flip(dims=(1,))
        input_mask = past_observed_values.flip(dims=(1,))
        forecast = past_target[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return loc + forecast * scale


class MetaNBEATSModel(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        model_type: str,
        trend_blocks: int = 3,
        trend_layers: int = 4,
        trend_layer_size: int = 256,
        degree_of_polynomial: int = 3,
        seasonality_blocks: int = 3,
        seasonality_layers: int = 4,
        seasonality_layer_size: int = 2048,
        num_of_harmonics: int = 1,
        stacks: int = 30,
        layers: int = 4,
        layer_size: int = 512,
        scale: bool = False,
    ):
        super().__init__()
        if model_type == "interpretable":
            self.model = NBEATSInterpretableModel(
                prediction_length=prediction_length,
                context_length=context_length,
                trend_blocks=trend_blocks,
                trend_layers=trend_layers,
                trend_layer_size=trend_layer_size,
                degree_of_polynomial=degree_of_polynomial,
                seasonality_blocks=seasonality_blocks,
                seasonality_layers=seasonality_layers,
                seasonality_layer_size=seasonality_layer_size,
                num_of_harmonics=num_of_harmonics,
                scale=scale,
            )
        elif model_type == "generic":
            self.model = NBEATSGenericModel(
                prediction_length=prediction_length,
                context_length=context_length,
                stacks=stacks,
                layers=layers,
                layer_size=layer_size,
                scale=scale,
            )
        else:
            raise ValueError(f"Unknown model type {model_type}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
