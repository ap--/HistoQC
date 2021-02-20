"""pep8 shim for histoqc.BrightContrastModule with pep484 type annotations"""
from typing import TYPE_CHECKING

from histoqc.BrightContrastModule import getBrightnessGray as _getBrightnessGray
from histoqc.BrightContrastModule import getBrightnessByChannelinColorSpace as _getBrightnessByChannelinColorSpace
from histoqc.BrightContrastModule import getContrast as _getContrast

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineState

__all__ = [
    'get_brightness_gray',
    'get_brightness_by_channel_in_color_space',
    'get_contrast',
]


def get_brightness_gray(
    pstate: PipelineState,
    *,
    limit_to_mask: bool = True
) -> np.ndarray:
    return pstate.call(_getBrightnessGray, limit_to_mask=str(limit_to_mask))


def get_brightness_by_channel_in_color_space(
    pstate: PipelineState,
    *,
    limit_to_mask: bool = True,
    to_color_space: str = "RGB",
) -> np.ndarray:
    return pstate.call(
        _getBrightnessByChannelinColorSpace,
        limit_to_mask=str(limit_to_mask),
        to_color_space=to_color_space,
    )


def get_contrast(
    pstate: PipelineState,
    *,
    limit_to_mask: bool = True,
) -> np.ndarray:
    return pstate.call(_getContrast, limit_to_mask=str(limit_to_mask))
