from typing import Type

import pytorch_lightning as pl

from .supervised import Supervised
from .meta_nbeats import MetaNBEATS


def get_pretrain_cls(name) -> Type[pl.LightningModule]:
    return {
        "supervised": Supervised,
        "meta_nbeats": MetaNBEATS,
    }[name]
