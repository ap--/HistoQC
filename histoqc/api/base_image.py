"""pep8 shim for histoqc.BaseImage with pep484 type annotations"""
import sys
from typing import Optional
from typing import TYPE_CHECKING

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from histoqc.BaseImage import getMag as _getMag

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable

__all__ = ['get_mag', 'MaskStatisticsType']

MaskStatisticsType = Literal["relative2mask", "absolute", "relative2image"]


def get_mag(pstate: PipelineCallable, *, confirm_base_mag: str = "False") -> Optional[np.ndarray]:
    _cbm = str(confirm_base_mag)
    return pstate.histoqc_call(_getMag, confirm_base_mag=_cbm)
