from dataclasses import dataclass

import hydra
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_dataset_builder
from gluonts.dataset.split import DateSplitter
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.modules .loss import (
    CRPS,
    NegativeLogLikelihood,
    QuantileLoss,
)
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from util.prepare import create_training_data_loader
from util.torch.quantile_output import QuantileOutput
from util.torch.distributions import (
    IndependentStudentTOutput,
    MultivariateStudentTOutput,
    SQFOutput,
    ISQFOutput,
    FlowOutput,
)
from .model.pretrain import get_pretrain_cls
from .model.backbone import get_backbone_cls


@dataclass
class Experiment:
    backbone_name: str
    size: str
    pretrain_name: str
    output_head: str
    output_dir: str
    backbone_cfg: dict
    pretrain_cfg: dict
    data_cfg: dict
    trainer_cfg: dict
    pretrained_ckpt: str
    seed: int = 42

    def __post_init__(self):
        pl.seed_everything(self.seed + rank_zero_only.rank, workers=True)
        data_path = "Salesforce/cloudops_tsf"
        dataset = load_dataset(
            path=data_path,
            name=self.data_cfg["dataset_name"],
            split="pretrain" if self.data_cfg["pretrain"] else "train_test",
        )
        self.dataset = dataset
        self.ds_config = load_dataset_builder(
            path=data_path,
            name=self.data_cfg["dataset_name"],
        ).config

    def __call__(self):
        pretrain_cfg = dict(self.pretrain_cfg)
        backbone_cfg = dict(self.backbone_cfg.copy())

        target_dim = self.ds_config.target_dim

        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if self.output_head == "student_t":
            if target_dim == 1:
                backbone_cfg["distr_output"] = StudentTOutput()
            else:
                backbone_cfg["distr_output"] = IndependentStudentTOutput(target_dim)
            pretrain_cfg["loss"] = NegativeLogLikelihood()
        elif self.output_head == "multivariate_student_t":
            assert target_dim > 1
            backbone_cfg["distr_output"] = MultivariateStudentTOutput(target_dim)
            pretrain_cfg["loss"] = NegativeLogLikelihood()
        elif self.output_head == "qf":
            backbone_cfg["distr_output"] = QuantileOutput(
                quantiles=quantiles, target_dim=target_dim
            )
            pretrain_cfg["loss"] = QuantileLoss()
        elif self.output_head == "iqf":
            backbone_cfg["distr_output"] = ISQFOutput(
                num_pieces=1, qk_x=[0.01, 0.1, 0.5, 0.9, 0.99], target_dim=target_dim
            )
            pretrain_cfg["loss"] = CRPS()
        elif self.output_head == "sqf":
            backbone_cfg["distr_output"] = SQFOutput(
                num_pieces=10, target_dim=target_dim
            )
            pretrain_cfg["loss"] = CRPS()
        elif self.output_head == "isqf":
            backbone_cfg["distr_output"] = ISQFOutput(
                num_pieces=3, qk_x=[0.01, 0.1, 0.5, 0.9, 0.99], target_dim=target_dim
            )
            pretrain_cfg["loss"] = CRPS()
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

        num_devices = (
            torch.cuda.device_count()
            if self.trainer_cfg["devices"] == "auto"
            else len(self.trainer_cfg["devices"])
            if isinstance(self.trainer_cfg["devices"], list)
            else self.trainer_cfg["devices"]
            if isinstance(self.trainer_cfg["devices"], int)
            else None  # raise error
        )
        loader_batch_size = self.data_cfg["batch_size"] // (
            num_devices * self.trainer_cfg["accumulate_grad_batches"]
        )

        model = get_pretrain_cls(self.pretrain_name)(
            backbone_cls=get_backbone_cls(self.backbone_name),
            backbone_args=backbone_cfg,
            freq=self.ds_config.freq,
            prediction_length=self.ds_config.prediction_length,
            static_dim=static_dim,
            static_cardinalities=static_cardinalities,
            past_dynamic_dim=past_dynamic_dim,
            scaling=True,
            num_train_steps=self.trainer_cfg["max_epochs"]
            * self.data_cfg["num_batches_per_epoch"],
            **pretrain_cfg,
        )

        if self.pretrained_ckpt:
            model.from_pretrained(self.pretrained_ckpt)

        data_loader = create_training_data_loader(
            model,
            self.dataset,
            DateSplitter(self.ds_config.test_split_date),
            batch_size=loader_batch_size,
            num_batches_per_epoch=self.data_cfg["num_batches_per_epoch"]
            * self.trainer_cfg["accumulate_grad_batches"],
            num_workers=self.data_cfg["num_workers"],
        )

        trainer = pl.Trainer(
            **self.trainer_cfg,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    mode="min",
                    verbose=True,
                    dirpath=self.output_dir,
                    save_top_k=-1,
                    every_n_epochs=50,
                ),
                pl.callbacks.LearningRateMonitor(
                    logging_interval="epoch",
                ),
            ],
        )

        trainer.fit(
            model=model,
            train_dataloaders=data_loader,
        )


@hydra.main(version_base="1.1", config_path="conf/", config_name="pretrain_exp")
def main(cfg: DictConfig):
    hydra_cfg: dict[str, any] = OmegaConf.to_container(HydraConfig.get())

    output_dir = hydra_cfg["runtime"]["output_dir"]
    backbone_name = hydra_cfg["runtime"]["choices"]["backbone"]
    size = hydra_cfg["runtime"]["choices"]["size"]
    pretrain_name = hydra_cfg["runtime"]["choices"]["pretrain"]

    backbone_cfg = dict(cfg["backbone"]) | dict(cfg["size"])
    pretrain_cfg = cfg["pretrain"]
    data_cfg = cfg["data"]
    trainer_cfg = cfg["trainer"]
    seed = cfg["seed"]

    if cfg["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    Experiment(
        backbone_name=backbone_name,
        size=size,
        pretrain_name=pretrain_name,
        output_head=cfg["output_head"],
        output_dir=output_dir,
        backbone_cfg=backbone_cfg,
        pretrain_cfg=pretrain_cfg,
        data_cfg=data_cfg,
        trainer_cfg=trainer_cfg,
        pretrained_ckpt=cfg["pretrained_ckpt"],
        seed=seed,
    )()


if __name__ == "__main__":
    main()
