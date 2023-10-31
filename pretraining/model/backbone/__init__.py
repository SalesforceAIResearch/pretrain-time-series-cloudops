from typing import Type

from torch import nn

from .encoder_decoder import CachedEncoderDecoderModel
from .encoder_decoder_dms import EncoderDecoderDMSModel
from .encoder import EncoderModel
from .masked_encoder import MaskedEncoderModel
from .one_fits_all import OneFitsAllModel
from .meta_nbeats import MetaNBEATSModel
from .ts2vec import TS2VecModel
from .cost import CoSTModel


def get_backbone_cls(name: str) -> Type[nn.Module]:
    if name.startswith("encoder_decoder_dms"):
        cls = EncoderDecoderDMSModel
    elif name.startswith("encoder_decoder"):
        cls = CachedEncoderDecoderModel
    elif name.startswith("encoder"):
        cls = EncoderModel
    elif name.startswith("masked_encoder"):
        cls = MaskedEncoderModel
    elif name.startswith("one_fits_all"):
        cls = OneFitsAllModel
    elif name.startswith("meta_nbeats"):
        cls = MetaNBEATSModel
    elif name.startswith("ts2vec"):
        cls = TS2VecModel
    elif name.startswith("cost"):
        cls = CoSTModel
    else:
        raise ValueError(f"Unknown backbone {name}")
    return cls
