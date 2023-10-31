from typing import Iterator, List, Optional

import numpy as np
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform._base import FlatMapTransformation
from gluonts.transform.sampler import InstanceSampler


class InstanceSplitter(FlatMapTransformation):
    """
    Selects training instances, by slicing the target and other time series
    like arrays at random points in training mode or at the last time point in
    prediction mode. Assumption is that all time like arrays start at the same
    time point.
    The target and each time_series_field is removed and instead two
    corresponding fields with prefix `past_` and `future_` are included. E.g.
    If the target array is one-dimensional, the resulting instance has shape
    (len_target). In the multi-dimensional case, the instance has shape (dim,
    len_target).
    target -> past_target and future_target
    The transformation also adds a field 'past_is_pad' that indicates whether
    values where padded or not.
    Convention: time axis is always the last axis.
    Parameters
    ----------
    instance_sampler
        instance sampler that provides sampling indices given a time series
    past_length
        length of the target seen before making prediction
    future_length
        length of the target that must be predicted
    target_field
        field containing the target (default: FieldName.TARGET)
    is_pad_field
        output field indicating whether padding happened (default: FieldName.IS_PAD)
    start_field
        field containing the start date of the time series (default: FieldName.START)
    forecast_start_field
        output field that will contain the time point where the forecast starts
        (default: FieldName.FORECAST_START)
    observed_value_field
        field containing the observed indicator (default: FieldName.OBSERVED_VALUES)
    time_series_fields
        fields that contains time series, they are split in the same interval
        as the target (default: None)
    past_time_series_fields
        fields that contains past time series, they are split in the same interval
        as the target, but future values are not included (default: None)
    lead_time
        gap between the past and future windows (default: 0)
    output_NTC
        whether to have time series output in (time, dimension) or in
        (dimension, time) layout (default: True)
    dummy_value
        Value to use for padding. (default: 0.0)
    """

    @validated()
    def __init__(
        self,
        instance_sampler: InstanceSampler,
        past_length: int,
        future_length: int,
        target_field: str = FieldName.TARGET,
        is_pad_field: str = FieldName.IS_PAD,
        start_field: str = FieldName.START,
        forecast_start_field: str = FieldName.FORECAST_START,
        observed_value_field: str = FieldName.OBSERVED_VALUES,
        time_field: Optional[str] = FieldName.FEAT_TIME,
        time_series_fields: Optional[List[str]] = None,
        past_time_series_fields: Optional[List[str]] = None,
        lead_time: int = 0,
        output_NTC: bool = True,
        dummy_value: float = 0.0,
    ) -> None:
        super().__init__()

        assert future_length > 0, "The value of `future_length` should be > 0"

        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.observed_value_field = observed_value_field
        self.time_field = time_field
        self.ts_fields = time_series_fields or []
        self.past_ts_fields = past_time_series_fields or []
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.dummy_value = dummy_value

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def _adjust_shape(self, arr: np.ndarray) -> np.ndarray:
        if self.output_NTC:
            return arr.transpose()
        return arr

    def flatmap_transform(self, data: DataEntry, is_train: bool) -> Iterator[DataEntry]:
        pl = self.future_length
        lt = self.lead_time
        target = data[self.target_field]

        sampled_indices = self.instance_sampler(target)

        slice_cols = set(
            self.ts_fields
            + self.past_ts_fields
            + [self.target_field, self.observed_value_field]
            + ([self.time_field] if self.time_field is not None else [])
        )

        for i in sampled_indices:
            pad_length = max(self.past_length - i, 0)
            d = data.copy()
            for field in slice_cols:
                # past piece
                if i >= self.past_length:
                    past_piece = d[field][..., i - self.past_length : i]
                else:
                    pad_block = np.full(
                        shape=d[field].shape[:-1] + (pad_length,),
                        fill_value=self.dummy_value,
                        dtype=d[field].dtype,
                    )
                    past_piece = np.concatenate([pad_block, d[field][..., :i]], axis=-1)

                # future piece
                future_piece = d[field][..., (i + lt) : (i + lt + pl)]

                if field in self.ts_fields:
                    # stay whole
                    future_piece = d[field][..., (i + lt) : (i + lt + pl)]
                    d[field] = self._adjust_shape(
                        np.concatenate([past_piece, future_piece], axis=-1)
                    )
                elif field in self.past_ts_fields:
                    # past only
                    d[field] = self._adjust_shape(past_piece)
                else:
                    # split
                    d[self._past(field)] = self._adjust_shape(past_piece)
                    d[self._future(field)] = self._adjust_shape(future_piece)
                    del d[field]

            pad_indicator = np.zeros(self.past_length, dtype=target.dtype)
            if pad_length > 0:
                pad_indicator[:pad_length] = 1

            d[self._past(self.is_pad_field)] = pad_indicator
            d[self.forecast_start_field] = d[self.start_field] + i + lt

            yield d
