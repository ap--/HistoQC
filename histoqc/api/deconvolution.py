"""pep8 shim for histoqc.DeconvolutionModule with pep484 type annotations"""
from typing import Literal
from typing import TYPE_CHECKING

from histoqc.DeconvolutionModule import seperateStains as _seperateStains

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineState

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
    pstate: PipelineState,
    *,
    stain: StainMethods,
    use_mask: bool = True,
) -> np.ndarray:
    return pstate.call(
        _seperateStains,
        stain=stain,
        use_mask=str(use_mask),
    )
