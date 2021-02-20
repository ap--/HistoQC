"""pep8 shim for histoqc.DeconvolutionModule with pep484 type annotations"""
from __future__ import annotations
import sys
from typing import Optional
from typing import TYPE_CHECKING

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from histoqc.DeconvolutionModule import seperateStains as _seperateStains

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable

__all__ = [
    'separate_stains',
]

StainMethods = Literal[
    "hed_from_rgb",
    "hdx_from_rgb",
    "fgx_from_rgb",
    "bex_from_rgb",
    "rbd_from_rgb",
    "gdx_from_rgb",
    "hax_from_rgb",
    "bro_from_rgb",
    "bpx_from_rgb",
    "ahx_from_rgb",
    "hpx_from_rgb",
]


def separate_stains(
    pstate: PipelineCallable,
    *,
    stain: StainMethods,
    use_mask: bool = True,
) -> Optional[np.ndarray]:
    return pstate.histoqc_call(
        _seperateStains,
        stain=stain,
        use_mask=str(use_mask),
    )
