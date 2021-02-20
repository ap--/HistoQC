"""pep8 shim for histoqc.BrightContrastModule with pep484 type annotations"""
from typing import Literal
from typing import TYPE_CHECKING

from histoqc.BrightContrastModule import getBrightnessGray as _getBrightnessGray
from histoqc.BrightContrastModule import getBrightnessByChannelinColorSpace as _getBrightnessByChannelinColorSpace
from histoqc.BrightContrastModule import getContrast as _getContrast

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable

__all__ = [
    'get_brightness_gray',
    'get_brightness_by_channel_in_color_space',
    'get_contrast',
]

# http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.convert_colorspace
ColorSpaceType = Literal[
    "RGB",
    "HSV",
    "RGB CIE",
    "XYZ",
    "YUV",
    "YIQ",
    "YPbPr",
    "YCbCr",
]


def get_brightness_gray(
    pstate: PipelineCallable,
    *,
    limit_to_mask: bool = True
) -> np.ndarray:
    return pstate.histoqc_call(_getBrightnessGray, limit_to_mask=str(limit_to_mask))


def get_brightness_by_channel_in_color_space(
    pstate: PipelineCallable,
    *,
    limit_to_mask: bool = True,
    to_color_space: ColorSpaceType = "RGB",
) -> np.ndarray:
    return pstate.histoqc_call(
        _getBrightnessByChannelinColorSpace,
        limit_to_mask=str(limit_to_mask),
        to_color_space=to_color_space,
    )


def get_contrast(
    pstate: PipelineCallable,
    *,
    limit_to_mask: bool = True,
) -> np.ndarray:
    return pstate.histoqc_call(_getContrast, limit_to_mask=str(limit_to_mask))
