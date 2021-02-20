"""pep8 shim for histoqc.BaseImage with pep484 type annotations"""
from typing import Literal
from typing import TYPE_CHECKING

from histoqc.BaseImage import getMag as _getMag

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineState

__all__ = ['get_mag', 'MaskStatisticsType']

MaskStatisticsType = Literal["relative2mask", "absolute", "relative2image"]


def get_mag(pstate: PipelineState, *, confirm_base_mag: str = "False") -> np.ndarray:
    _cbm = str(confirm_base_mag)
    return pstate.call(_getMag, confirm_base_mag=_cbm)
