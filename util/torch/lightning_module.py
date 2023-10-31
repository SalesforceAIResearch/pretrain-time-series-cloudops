from functools import cached_property
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import select
from gluonts.time_feature import (
    TimeFeature,
    time_features_from_frequency_str,
    get_lags_for_frequency,
)
from gluonts.torch.model.lightning_util import has_validation_loop
from gluonts.transform import (
    Transformation,
    RemoveFields,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    Chain,
)


class LightningModule(pl.LightningModule):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_shape: tuple[int, ...],
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
        age_feature: bool = True,
        lags_seq: Optional[list[int]] = None,
        scaling: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        patience: int = 10,
        forecast_type: str = "sample",
    ):
        super().__init__()
        self.freq = freq
        self.prediction_length = prediction_length
        self.target_shape = target_shape
        self.context_length = context_length or prediction_length

        self.static_dim = static_dim or 0
        self.dynamic_dim = dynamic_dim or 0
        self.past_dynamic_dim = past_dynamic_dim or 0
        self.static_cardinalities = static_cardinalities or []
        self.dynamic_cardinalities = dynamic_cardinalities or []
        self.past_dynamic_cardinalities = past_dynamic_cardinalities or []
        self.static_embedding_dim = (
            static_embedding_dim or []
            if static_embedding_dim is not None or static_cardinalities is None
            else [min(50, (cat + 1) // 2) for cat in static_cardinalities]
        )
        self.dynamic_embedding_dim = (
            dynamic_embedding_dim or []
            if dynamic_embedding_dim is not None or dynamic_cardinalities is None
            else [min(50, (cat + 1) // 2) for cat in dynamic_cardinalities]
        )
        self.past_dynamic_embedding_dim = (
            past_dynamic_embedding_dim or []
            if past_dynamic_embedding_dim is not None
            or past_dynamic_cardinalities is None
            else [min(50, (cat + 1) // 2) for cat in past_dynamic_cardinalities]
        )
        if time_features is None:
            self.time_features = time_features_from_frequency_str(freq)
        elif isinstance(time_features, list):
            self.time_features = time_features
        elif time_features == "none":
            self.time_features = []
        else:
            raise ValueError(
                f"Unknown value for time_features: {time_features}"
            )
        self.age_feature = age_feature
        self.time_dim = len(self.time_features) + int(age_feature)
        self.lags_seq = (
            get_lags_for_frequency(freq_str=freq, lag_ub=1200, num_lags=None)
            if lags_seq is None
            else lags_seq
        )

        self.scaling = scaling
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

        self.forecast_type = forecast_type

    def create_transformation(self) -> Transformation:
        remove_field_names = []
        if self.static_dim == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if self.dynamic_dim == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if self.past_dynamic_dim == 0:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
        if len(self.static_cardinalities) == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)
        if len(self.dynamic_cardinalities) == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_CAT)
        if len(self.past_dynamic_cardinalities) == 0:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_CAT)

        transforms = [
            RemoveFields(field_names=remove_field_names),
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1 + len(self.target_shape),
                dtype=np.float32,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
        ]

        if self.time_features:
            transforms += [
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                )
            ]

        if self.age_feature:
            transforms += [
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    if self.time_features
                    else [FieldName.FEAT_AGE],
                ),
            ]

        if self.time_features or self.age_feature:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_TIME,
                    expected_ndim=2,
                    dtype=np.float32,
                )
            ]

        if self.static_dim > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=np.float32,
                )
            ]

        if self.dynamic_dim > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_REAL,
                    expected_ndim=2,
                    dtype=np.int64,
                )
            ]

        if self.past_dynamic_dim > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    expected_ndim=2,
                    dtype=np.float32,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    output_field=f"observed_{FieldName.PAST_FEAT_DYNAMIC_REAL}",
                ),
                RemoveFields(
                    field_names=[f"observed_{FieldName.PAST_FEAT_DYNAMIC_REAL}"]
                ),
            ]

        if len(self.static_cardinalities) > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=np.int64
                )
            ]

        if len(self.dynamic_cardinalities) > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.FEAT_DYNAMIC_CAT, expected_ndim=2, dtype=np.int64
                )
            ]

        if len(self.past_dynamic_cardinalities) > 0:
            transforms += [
                AsNumpyArray(
                    field=FieldName.PAST_FEAT_DYNAMIC_CAT,
                    expected_ndim=2,
                    dtype=np.int64,
                )
            ]

        return Chain(transforms)

    def input_info(
        self, batch_size: int = 1
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        info = {
            "past_target": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
            "past_observed_values": (
                (batch_size, self.past_length) + self.target_shape,
                torch.float,
            ),
        }
        if self.time_dim > 0:
            info["past_time_feat"] = (
                (batch_size, self.past_length, self.time_dim),
                torch.float,
            )
            info["future_time_feat"] = (
                (batch_size, self.prediction_length, self.time_dim),
                torch.float,
            )
        if self.static_dim > 0:
            info["feat_static_real"] = (
                (batch_size, self.static_dim),
                torch.float,
            )
        if self.dynamic_dim > 0:
            info["feat_dynamic_real"] = (
                (
                    batch_size,
                    self.past_length + self.prediction_length,
                    self.dynamic_dim,
                ),
                torch.float,
            )
        if self.past_dynamic_dim > 0:
            info["past_feat_dynamic_real"] = (
                (batch_size, self.past_length, self.past_dynamic_dim),
                torch.float,
            )
        if len(self.static_cardinalities) > 0:
            info["feat_static_cat"] = (
                (batch_size, len(self.static_cardinalities)),
                torch.long,
            )
        if len(self.dynamic_cardinalities) > 0:
            info["feat_dynamic_cat"] = (
                (
                    batch_size,
                    self.past_length + self.prediction_length,
                    len(self.dynamic_cardinalities),
                ),
                torch.long,
            )
        if len(self.past_dynamic_cardinalities) > 0:
            info["past_feat_dynamic_cat"] = (
                (batch_size, self.past_length, len(self.past_dynamic_cardinalities)),
                torch.long,
            )
        return info

    @cached_property
    def past_length(self) -> int:
        lags = max(self.lags_seq) if self.lags_seq else 0
        return self.context_length + lags

    @cached_property
    def training_input_names(self) -> list[str]:
        return list(
            ["future_target", "future_observed_values"] + self.prediction_input_names
        )

    @cached_property
    def prediction_input_names(self) -> list[str]:
        return list(self.input_info().keys())

    @property
    def example_input_array(self) -> dict[str, torch.Tensor]:
        return {
            name: (
                torch.ones(shape, dtype=dtype)
                if "observed" in name
                else torch.zeros(shape, dtype=dtype)
            )
            for name, (shape, dtype) in self.input_info().items()
        }

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Execute training step.
        :param batch:
        :param batch_idx:
        :return:
        """
        train_loss = self.loss(
            **select(self.training_input_names, batch, ignore_missing=True),
        )

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Execute validation step.
        :param batch:
        :param batch_idx:
        :return:
        """
        val_loss = self.loss(
            **select(self.training_input_names, batch, ignore_missing=True),
        )

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        return val_loss

    def configure_optimizers(self):
        """
        Returns the optimizer to use.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        monitor = "val_loss" if has_validation_loop(self.trainer) else "train_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=0.5,
                    patience=self.patience,
                ),
                "monitor": monitor,
            },
        }
