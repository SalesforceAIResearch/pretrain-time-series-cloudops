from typing import Optional

import numpy as np
import pytorch_lightning as pl
from datasets import Dataset
from gluonts.model.forecast_generator import (
    DistributionForecastGenerator,
)
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import DateSplitter, OffsetSplitter
from gluonts.time_feature import (
    minute_of_hour,
    hour_of_day,
    day_of_week,
    day_of_month,
    day_of_year,
    week_of_year,
    month_of_year,
)
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    AsNumpyArray,
    Transformation,
    Chain,
    RemoveFields,
    SelectFields,
    TestSplitSampler,
)
from torch.utils.data import DataLoader

from .dataset import TransformedIterableDataset
from .transform.sampler import ValidationRegionSampler, RandomSampler
from .transform.convert import ProcessDataEntryTransform
from .transform.split import InstanceSplitter
from .torch.lightning_module import LightningModule
from util.forecast_generator import SampleForecastGenerator, QuantileForecastGenerator


def create_transformation(
    target_dim: Optional[int] = None,
    time_feat: bool = False,
    age_feat: bool = False,
    prediction_length: int = 0,
) -> Transformation:
    remove_field_names = [
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_DYNAMIC_REAL,
        FieldName.PAST_FEAT_DYNAMIC_CAT,
        FieldName.PAST_FEAT_DYNAMIC_REAL,
    ]
    transforms = [RemoveFields(field_names=remove_field_names)]

    if time_feat:
        transforms += [
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=[
                    minute_of_hour,
                    hour_of_day,
                    day_of_week,
                    day_of_month,
                    day_of_year,
                    week_of_year,
                    month_of_year,
                ],
                pred_length=prediction_length,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_TIME,
                expected_ndim=2,
                dtype=np.float32,
            ),
        ]

    if age_feat:
        transforms += [
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=False,
            ),
            AsNumpyArray(
                field=FieldName.FEAT_AGE,
                expected_ndim=2,
                dtype=np.float32,
            ),
        ]

    transforms += [
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )
    ]

    if target_dim is not None:
        transforms += [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1 if target_dim == 1 else 2,
                dtype=np.float32,
            )
        ]

    return Chain(transforms)


def create_instance_splitter(
    module: pl.LightningModule,
    mode: str,
    num_validation_instances: int = 1,
) -> InstanceSplitter:
    assert mode in ["training", "validation", "test"]

    instance_sampler = {
        "training": RandomSampler(
            min_past=1,
            min_future=module.prediction_length,
        ),
        "validation": ValidationRegionSampler(
            num_instances=num_validation_instances, min_future=module.prediction_length
        ),
        "test": TestSplitSampler(),
    }[mode]

    ts_fields = []
    if module.dynamic_dim > 0:
        ts_fields.append(FieldName.FEAT_DYNAMIC_REAL)
    if len(module.dynamic_cardinalities) > 0:
        ts_fields.append(FieldName.FEAT_DYNAMIC_CAT)

    past_ts_fields = []
    if module.past_dynamic_dim > 0:
        past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)
    if len(module.past_dynamic_cardinalities) > 0:
        past_ts_fields.append(FieldName.PAST_FEAT_DYNAMIC_CAT)

    return InstanceSplitter(
        instance_sampler=instance_sampler,
        past_length=module.past_length,
        future_length=module.prediction_length,
        time_field=FieldName.FEAT_TIME if module.time_dim > 0 else None,
        time_series_fields=ts_fields,
        past_time_series_fields=past_ts_fields,
    )


def create_training_data_loader(
    module: LightningModule,
    data: Dataset,
    dataset_splitter: DateSplitter | OffsetSplitter,
    batch_size: int,
    num_batches_per_epoch: int,
    num_workers: int,
    allow_missing: bool = False,
) -> DataLoader:
    transformation = (
        module.create_transformation()
        + create_instance_splitter(module, "training")
        + SelectFields(module.training_input_names, allow_missing=allow_missing)
    )

    dataset = TransformedIterableDataset(
        data,
        ProcessDataEntryTransform(module.freq, len(module.target_shape) == 0),
        transformation,
        splitter=dataset_splitter,
        is_train=True,
        sample="proportional",
        num_batches_per_epoch=num_batches_per_epoch,
        batch_size=batch_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=dataset.worker_init_fn,
        persistent_workers=True,
    )


def create_validation_data_loader(
    module: LightningModule,
    data: Dataset,
    dataset_splitter: DateSplitter | OffsetSplitter,
    batch_size: int,
    num_workers: int,
    allow_missing: bool = False,
) -> DataLoader:
    transformation = (
        module.create_transformation()
        + create_instance_splitter(module, "validation")
        + SelectFields(module.training_input_names, allow_missing=allow_missing)
    )
    dataset = TransformedIterableDataset(
        data,
        ProcessDataEntryTransform(module.freq, len(module.target_shape) == 0),
        transformation,
        splitter=dataset_splitter,
        is_train=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=dataset.worker_init_fn,
        persistent_workers=True,
    )


def create_predictor(
    module: LightningModule,
    batch_size: int,
) -> PyTorchPredictor:
    transformation = module.create_transformation() + create_instance_splitter(
        module, "test"
    )

    if module.forecast_type == "sample":
        forecast_generator = SampleForecastGenerator()
    elif module.forecast_type == "quantile":
        forecast_generator = QuantileForecastGenerator(
            quantiles=[str(q) for q in module.quantiles],
        )
    elif module.forecast_type == "distribution":
        forecast_generator = DistributionForecastGenerator(
            distribution=module.distr_output,
        )
    else:
        raise ValueError(f"Unknown forecast_type: {module.forecast_type}")

    return PyTorchPredictor(
        input_transform=transformation,
        input_names=module.prediction_input_names,
        prediction_net=module,
        batch_size=batch_size,
        prediction_length=module.prediction_length,
        device=module.device,
        forecast_generator=forecast_generator,
    )
