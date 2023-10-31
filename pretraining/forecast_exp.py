from dataclasses import dataclass
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_dataset_builder
from gluonts.dataset.split import OffsetSplitter, split
from gluonts.dataset.common import ProcessDataEntry
from gluonts.itertools import Map
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.modules .loss import (
    CRPS,
    NegativeLogLikelihood,
    QuantileLoss,
)
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter

from util.evaluation import evaluate_model, get_metrics
from util.prepare import create_training_data_loader, create_predictor
from util.torch.quantile_output import QuantileOutput
from util.torch.distributions import (
    IndependentStudentTOutput,
    MultivariateStudentTOutput,
    SQFOutput,
    ISQFOutput,
    FlowOutput,
)
from .model.forecast import get_forecast_cls
from .model.backbone import get_backbone_cls


@dataclass
class Experiment:
    backbone_name: str
    backbone_cfg: dict
    forecast_name: str
    forecast_cfg: dict
    output_head: str
    output_dir: str
    data_cfg: dict
    trainer_cfg: dict
    pretrained_ckpt: str
    test: bool = False
    seed: int = 42
    ckpt_epoch: Optional[int] = None

    def __post_init__(self):
        pl.seed_everything(self.seed + rank_zero_only.rank)

        self.dataset = load_dataset(
            path="Salesforce/cloudops_tsf",
            name=self.data_cfg["dataset_name"],
            split="train_test",
        )

        self.ds_config = load_dataset_builder(
            path="Salesforce/cloudops_tsf",
            name=self.data_cfg["dataset_name"],
        ).config

        if self.test:
            self.train_offset = -(
                self.ds_config.prediction_length
                + self.ds_config.stride * (self.ds_config.rolling_evaluations - 1)
            )
            self.validation_offset = None
        else:
            self.train_offset = -(
                2 * self.ds_config.prediction_length
                + self.ds_config.stride * (self.ds_config.rolling_evaluations - 1)
            )
            self.validation_offset = None

    def get_params(self) -> tuple[dict, dict]:
        forecast_cfg = dict(self.forecast_cfg)
        backbone_cfg = dict(self.backbone_cfg.copy())

        target_dim = self.ds_config.target_dim
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if self.output_head == "student_t":
            if target_dim == 1:
                backbone_cfg["distr_output"] = StudentTOutput()
            else:
                backbone_cfg["distr_output"] = IndependentStudentTOutput(target_dim)
            forecast_cfg["loss"] = NegativeLogLikelihood()
        elif self.output_head == "multivariate_student_t":
            assert target_dim > 1
            backbone_cfg["distr_output"] = MultivariateStudentTOutput(target_dim)
            forecast_cfg["loss"] = NegativeLogLikelihood()
        elif self.output_head == "qf":
            backbone_cfg["distr_output"] = QuantileOutput(
                quantiles=quantiles, target_dim=target_dim
            )
            forecast_cfg["loss"] = QuantileLoss()
            forecast_cfg["forecast_type"] = "quantile"
        elif self.output_head == "iqf":
            backbone_cfg["distr_output"] = ISQFOutput(
                num_pieces=1, qk_x=[0.01, 0.1, 0.5, 0.9, 0.99], target_dim=target_dim
            )
            forecast_cfg["loss"] = CRPS()
            forecast_cfg["forecast_type"] = "quantile"
        elif self.output_head == "sqf":
            backbone_cfg["distr_output"] = SQFOutput(
                num_pieces=10, target_dim=target_dim
            )
            forecast_cfg["loss"] = CRPS()
            forecast_cfg["forecast_type"] = "quantile"
        elif self.output_head == "isqf":
            backbone_cfg["distr_output"] = ISQFOutput(
                num_pieces=3, qk_x=[0.01, 0.1, 0.5, 0.9, 0.99], target_dim=target_dim
            )
            forecast_cfg["loss"] = CRPS()
            forecast_cfg["forecast_type"] = "quantile"
        elif self.output_head in ("real_nvp", "maf"):
            backbone_cfg["distr_output"] = FlowOutput(
                flow=self.output_head,
                input_size=target_dim,
                cond_size=200,
                n_blocks=3,
                hidden_size=backbone_cfg["d_model"],
                n_hidden=2,
            )
        elif self.output_head is None:
            pass
        else:
            raise ValueError(f"Unknown output head {self.output_head}")
        
        static_cardinalities = None
        static_dim = self.ds_config.feat_static_real_dim
        past_dynamic_dim = self.ds_config.past_feat_dynamic_real_dim
        forecast_cfg |= {
            "static_dim": static_dim,
            "static_cardinalities": static_cardinalities,
            "past_dynamic_dim": past_dynamic_dim,
        }
        return backbone_cfg, forecast_cfg

    def __call__(self):
        backbone_cfg, forecast_cfg = self.get_params()

        model = get_forecast_cls(self.forecast_name)(
            backbone_cls=get_backbone_cls(self.backbone_name),
            backbone_args=backbone_cfg,
            freq=self.ds_config.freq,
            prediction_length=self.ds_config.prediction_length,
            scaling=True,
            **forecast_cfg,
        )

        if self.pretrained_ckpt:
            model.from_pretrained(self.pretrained_ckpt)

        if self.forecast_name == "finetune":
            trainer_cfg = dict(self.trainer_cfg)

            train_data_loader = create_training_data_loader(
                model,
                self.dataset,
                OffsetSplitter(self.train_offset),
                batch_size=self.data_cfg["batch_size"],
                num_batches_per_epoch=self.data_cfg["num_batches_per_epoch"],
                num_workers=self.data_cfg["num_workers"],
            )

            checkpoint = pl.callbacks.ModelCheckpoint(
                monitor="train_loss", mode="min", verbose=True, dirpath=self.output_dir
            )
            trainer = pl.Trainer(
                **trainer_cfg,
                callbacks=[
                    checkpoint,
                    pl.callbacks.EarlyStopping(
                        monitor="train_loss", patience=10, mode="min"
                    ),
                ],
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_data_loader,
            )
            model = model.load_from_checkpoint(checkpoint.best_model_path)

        predictor = create_predictor(
            model.to("cuda"),
            batch_size=self.data_cfg["sampling_batch_size"],
        )

        process = ProcessDataEntry(
            self.ds_config.freq,
            one_dim_target=self.ds_config.univariate,
            use_timestamp=False,
        )
        dataset = Map(process, self.dataset)
        _, test_template = split(dataset, offset=self.train_offset)
        test_data = test_template.generate_instances(
            self.ds_config.prediction_length,
            windows=(self.ds_config.rolling_evaluations if self.test else 1),
            distance=self.ds_config.stride,
        )

        metrics, agg_metrics = get_metrics(univariate=self.ds_config.univariate)

        if self.trainer_cfg["precision"] in ("bf16", "bf16-mixed"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                results = evaluate_model(
                    predictor,
                    test_data=test_data,
                    metrics=metrics,
                    agg_metrics=agg_metrics,
                    axis=None,
                    seasonality=1,
                )
        else:
            results = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                agg_metrics=agg_metrics,
                axis=None,
                seasonality=1,
            )

        results = {k: v[0] for k, v in results.to_dict("list").items()}
        writer = SummaryWriter("lightning_logs/version_0")
        for k, v in results.items():
            writer.add_scalar(k, v)
        writer.close()


@hydra.main(version_base="1.1", config_path="conf/", config_name="forecast_exp")
def main(cfg: DictConfig):
    hydra_cfg: dict[str, any] = OmegaConf.to_container(HydraConfig.get())

    output_dir = hydra_cfg["runtime"]["output_dir"]
    backbone_name = (
        f"{hydra_cfg['runtime']['choices']['backbone']}"
        f"_{hydra_cfg['runtime']['choices']['size']}"
    )
    forecast_name = hydra_cfg["runtime"]["choices"]["forecast"]

    backbone_cfg = dict(cfg["backbone"]) | dict(cfg["size"])
    forecast_cfg = cfg["forecast"]
    data_cfg = cfg["data"]
    trainer_cfg = cfg["trainer"]
    test = cfg["test"]
    seed = cfg["seed"]
    pretrained_ckpt = cfg["pretrained_ckpt"]

    assert forecast_name in ["finetune", "zeroshot", "meta_nbeats"]

    if cfg["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    experiment = Experiment(
        backbone_name=backbone_name,
        backbone_cfg=backbone_cfg,
        forecast_name=forecast_name,
        forecast_cfg=forecast_cfg,
        output_head=cfg["output_head"],
        output_dir=output_dir,
        data_cfg=data_cfg,
        trainer_cfg=trainer_cfg,
        pretrained_ckpt=pretrained_ckpt,
        test=test,
        seed=seed,
        ckpt_epoch=cfg["ckpt_epoch"],
    )
    experiment()


if __name__ == "__main__":
    main()
