from typing import Type

import pytorch_lightning as pl

from .finetune import Finetune
from .zeroshot import Zeroshot
from .meta_nbeats import NBEATSEnsemble


def get_forecast_cls(name) -> Type[pl.LightningModule]:
    return {
        "finetune": Finetune,
        "zeroshot": Zeroshot,
        "meta_nbeats": NBEATSEnsemble,
    }[name]
