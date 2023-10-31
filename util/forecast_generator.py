from typing import Callable, Iterator, Optional

import numpy as np

from torch import nn
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import select
from gluonts.dataset.loader import DataLoader
from gluonts.model.forecast import Forecast, SampleForecast, QuantileForecast as _QuantileForecast
from gluonts.model.forecast_generator import (
    OutputTransform,
    ForecastGenerator,
    log_once,
    NOT_SAMPLE_BASED_MSG,
)


class QuantileForecast(_QuantileForecast):
    def copy_aggregate(self, agg_fun: Callable) -> "QuantileForecast":
        if len(self.forecast_array.shape) == 2:
            forecast_arrays = self.forecast_array
        else:
            forecast_arrays = agg_fun(self.forecast_array, axis=2)

        return QuantileForecast(
            forecast_arrays=forecast_arrays,
            start_date=self.start_date,
            forecast_keys=self.forecast_keys,
            item_id=self.item_id,
            info=self.info,
        )


def predict_to_numpy(prediction_net: nn.Module, kwargs) -> np.ndarray:
    return prediction_net(**kwargs).cpu().numpy()


class SampleForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self):
        pass

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: list[str],
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = select(input_names, batch, ignore_missing=True)
            outputs = predict_to_numpy(prediction_net, inputs)
            if output_transform is not None:
                outputs = output_transform(batch, outputs)
            if num_samples:
                num_collected_samples = outputs[0].shape[0]
                collected_samples = [outputs]
                while num_collected_samples < num_samples:
                    outputs = predict_to_numpy(prediction_net, inputs)
                    if output_transform is not None:
                        outputs = output_transform(batch, outputs)
                    collected_samples.append(outputs)
                    num_collected_samples += outputs[0].shape[0]
                outputs = [
                    np.concatenate(s)[:num_samples] for s in zip(*collected_samples)
                ]
                assert len(outputs[0]) == num_samples
            i = -1

            for i, output in enumerate(outputs):
                yield SampleForecast(
                    output,
                    start_date=batch[FieldName.FORECAST_START][i],
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                )
            assert i + 1 == len(batch[FieldName.FORECAST_START])


class QuantileForecastGenerator(ForecastGenerator):
    @validated()
    def __init__(self, quantiles: list[str]) -> None:
        self.quantiles = quantiles

    def __call__(
        self,
        inference_data_loader: DataLoader,
        prediction_net,
        input_names: list[str],
        output_transform: Optional[OutputTransform],
        num_samples: Optional[int],
        **kwargs
    ) -> Iterator[Forecast]:
        for batch in inference_data_loader:
            inputs = select(input_names, batch, ignore_missing=True)
            outputs = predict_to_numpy(prediction_net, inputs)
            if output_transform is not None:
                outputs = output_transform(batch, outputs)

            if num_samples:
                log_once(NOT_SAMPLE_BASED_MSG)

            i = -1
            for i, output in enumerate(outputs):
                yield QuantileForecast(
                    output,
                    start_date=batch[FieldName.FORECAST_START][i],
                    item_id=batch[FieldName.ITEM_ID][i]
                    if FieldName.ITEM_ID in batch
                    else None,
                    info=batch["info"][i] if "info" in batch else None,
                    forecast_keys=self.quantiles,
                )
            assert i + 1 == len(batch[FieldName.FORECAST_START])
