import logging
from collections import ChainMap
from dataclasses import dataclass, InitVar
from functools import partial
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.split import TestData
from gluonts.ev import (
    absolute_error,
    Metric,
    seasonal_error,
    MAPE,
    MASE,
    MSE,
    MSIS,
    SMAPE,
    Coverage,
    SumAbsoluteError,
    SumQuantileLoss,
    mean_absolute_label,
    sum_absolute_label,
    ND,
    NRMSE,
    RMSE,
    MAECoverage,
    MeanSumQuantileLoss,
    MeanWeightedSumQuantileLoss,
    WeightedSumQuantileLoss,
    DirectEvaluator,
    Mean,
)
from gluonts.gluonts_tqdm import tqdm
from gluonts.itertools import prod
from gluonts.model import Forecast, Predictor
from gluonts.model.forecast import Quantile, SampleForecast
from gluonts.time_feature.seasonality import get_seasonality
from toolz import first, valmap

logger = logging.getLogger(__name__)


def get_metrics(
    quantiles: Optional[tuple[str]] = None,
    univariate: bool = True,
) -> tuple[list[Metric], list[tuple[Metric, Callable]]]:
    if quantiles is None:
        quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    quantiles = tuple(map(Quantile.parse, quantiles))

    univariate_metrics = [
        sum_absolute_label,
        SumAbsoluteError(),
        *(SumQuantileLoss(q=quantile.value) for quantile in quantiles),
        mean_absolute_label,
        MSE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        *(Coverage(q=quantile.value) for quantile in quantiles),
        RMSE(),
        NRMSE(),
        ND(),
        *(WeightedSumQuantileLoss(q=quantile.value) for quantile in quantiles),
        MeanSumQuantileLoss([quantile.value for quantile in quantiles]),
        MeanWeightedSumQuantileLoss([quantile.value for quantile in quantiles]),
        MAECoverage([quantile.value for quantile in quantiles]),
    ]

    if univariate:
        return univariate_metrics, []

    multivariate_metrics = [
        (
            MeanWeightedSumQuantileLoss([quantile.value for quantile in quantiles]),
            np.sum,
        )  # CRPS_sum
    ]

    return univariate_metrics, multivariate_metrics


@dataclass
class MAE:
    """Mean Absolute Error"""

    forecast_type: str = "mean"

    def __call__(self, axis: Optional[int] = None) -> DirectEvaluator:
        return DirectEvaluator(
            name="MAE",
            stat=partial(absolute_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )


@dataclass
class BatchForecast:
    """
    Wrapper around ``Forecast`` objects, that adds a batch dimension
    to arrays returned by ``__getitem__``, for compatibility with
    ``gluonts.ev``.
    """

    forecast: Forecast
    allow_nan: bool = False
    agg_fn: InitVar[Optional[Callable]] = None

    def __post_init__(self, agg_fn: Optional[Callable] = None):
        if agg_fn is not None:
            self.forecast = self.forecast.copy_aggregate(agg_fn)

    def __getitem__(self, name):
        value = self.forecast[name]
        if np.isnan(value).any():
            if name == "mean" and isinstance(self.forecast, SampleForecast):
                value = np.ma.masked_invalid(self.forecast.samples).mean(axis=0).data
            else:
                if not self.allow_nan:
                    raise ValueError("Forecast contains NaN values")

                logger.warning(
                    "Forecast contains NaN values. Metrics may be incorrect."
                )

        return np.expand_dims(value.T, axis=0)


def _get_data_batch(
    input_: DataEntry,
    label: DataEntry,
    forecast: Forecast,
    seasonality: Optional[int] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    agg_fn: Optional[Callable] = None,
) -> dict:
    forecast_dict = BatchForecast(forecast, allow_nan=allow_nan_forecast, agg_fn=agg_fn)

    freq = forecast.start_date.freqstr
    if seasonality is None:
        seasonality = get_seasonality(freq=freq)

    label_target = label["target"]
    input_target = input_["target"]

    if agg_fn is not None:
        label_target = agg_fn(label_target, axis=0)
        input_target = agg_fn(input_target, axis=0)

    if mask_invalid_label:
        label_target = np.ma.masked_invalid(label_target)
        input_target = np.ma.masked_invalid(input_target)

    other_data = {
        "label": np.expand_dims(label_target, axis=0),
        "seasonal_error": np.expand_dims(
            seasonal_error(input_target, seasonality=seasonality, time_axis=-1),
            axis=0,
        ),
    }

    return ChainMap(other_data, forecast_dict)


def evaluate_forecasts_raw(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    agg_metrics: Optional[list[tuple[Metric, Callable]]] = None,
    axis: Optional[Union[int, tuple]] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
    num_series: Optional[int] = None,
) -> dict:
    """
    Evaluate ``forecasts`` by comparing them with ``test_data``, according
    to ``metrics``.
    .. note:: This feature is experimental and may be subject to changes.
    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the time dimension
    - ``2`` aggregates across the target dimension (multivariate setting)
    Return results as a dictionary.
    """
    label_ndim = first(test_data.label)["target"].ndim

    assert label_ndim in [1, 2]
    assert axis is None or axis in [0, 1, 2]

    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    agg_evaluators = {}
    agg_metrics = agg_metrics or dict()
    for metric, agg_fn in agg_metrics:
        evaluator = metric(axis=axis)
        agg_evaluators[f"{agg_fn.__name__}_{evaluator.name}"] = (evaluator, agg_fn)

    index_data = []

    for input_, label, forecast in tqdm(
        zip(test_data.input, test_data.label, forecasts),
        total=num_series,
        desc="Running evaluation",
    ):
        if axis != 0:
            index_data.append((forecast.item_id, forecast.start_date))

        data_batch = _get_data_batch(
            input_,
            label,
            forecast,
            seasonality=seasonality,
            mask_invalid_label=mask_invalid_label,
            allow_nan_forecast=allow_nan_forecast,
        )

        for evaluator in evaluators.values():
            evaluator.update(data_batch)

        for evaluator, agg_fn in agg_evaluators.values():
            evaluator.update(
                _get_data_batch(
                    input_,
                    label,
                    forecast,
                    seasonality=seasonality,
                    mask_invalid_label=mask_invalid_label,
                    allow_nan_forecast=allow_nan_forecast,
                    agg_fn=agg_fn,
                )
            )

    metrics_values = {
        metric_name: evaluator.get() for metric_name, evaluator in evaluators.items()
    } | {
        metric_name: evaluator.get()
        for metric_name, (evaluator, agg_fn) in agg_evaluators.items()
    }

    if index_data:
        metrics_values["__index_0"] = index_data

    return metrics_values


def evaluate_forecasts(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    agg_metrics: Optional[list[tuple[Metric, Callable]]] = None,
    axis: Optional[Union[int, tuple]] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
    num_series: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate ``forecasts`` by comparing them with ``test_data``, according
    to ``metrics``.
    .. note:: This feature is experimental and may be subject to changes.
    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the time dimension
    - ``2`` aggregates across the target dimension (multivariate setting)
    Return results as a Pandas ``DataFrame``.
    """
    metrics_values = evaluate_forecasts_raw(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        agg_metrics=agg_metrics,
        axis=axis,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
        num_series=num_series,
    )
    index0 = metrics_values.pop("__index_0", None)

    metric_shape = metrics_values[first(metrics_values)].shape
    if metric_shape == ():
        index = [None]
    else:
        index_arrays = np.unravel_index(range(prod(metric_shape)), metric_shape)
        if index0 is not None:
            index0_repeated = np.take(index0, indices=index_arrays[0], axis=0)
            index_arrays = (*zip(*index0_repeated), *index_arrays[1:])
        index = pd.MultiIndex.from_arrays(index_arrays)

    flattened_metrics = valmap(np.ravel, metrics_values)

    return pd.DataFrame(flattened_metrics, index=index)


def evaluate_model(
    model: Predictor,
    *,
    test_data: TestData,
    metrics,
    agg_metrics: Optional[list[tuple[Metric, Callable]]] = None,
    axis: Optional[Union[int, tuple]] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate ``model`` when applied to ``test_data``, according
    to ``metrics``.
    .. note:: This feature is experimental and may be subject to changes.
    The optional ``axis`` arguments controls aggregation of the metrics:
    - ``None`` (default) aggregates across all dimensions
    - ``0`` aggregates across the dataset
    - ``1`` aggregates across the time dimension
    - ``2`` aggregates across the target dimension (multivariate setting)
    Return results as a Pandas ``DataFrame``.
    """
    forecasts = model.predict(test_data.input)

    return evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        agg_metrics=agg_metrics,
        axis=axis,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )
