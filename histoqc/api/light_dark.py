"""pep8 shim for histoqc.LightDarkModule with pep484 type annotations"""
from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.LightDarkModule import getIntensityThresholdOtsu as _getIntensityThresholdOtsu
from histoqc.LightDarkModule import getIntensityThresholdPercent as _getIntensityThresholdPercent

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable
    from .base_image import MaskStatisticsType

__all__ = [
    'get_intensity_threshold_otsu',
    'get_intensity_threshold_percent',
]


def get_intensity_threshold_otsu(
    pstate: PipelineCallable,
    *,
    local: bool = False,
    radius: float = 15.0,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> Optional[np.ndarray]:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _getIntensityThresholdOtsu,
        local=str(local),
        radius=radius,
        **extra,
    )


def get_intensity_threshold_percent(
    pstate: PipelineCallable,
    *,
    name: str = "classTask",
    lower_threshold: float = float("-inf"),
    upper_threshold: float = float("inf"),
    lower_variance: float = float("-inf"),
    upper_variance: float = float("inf"),
    invert: bool = False,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> Optional[np.ndarray]:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _getIntensityThresholdPercent,
        name=name,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        lower_variance=lower_variance,
        upper_variance=upper_variance,
        invert=str(invert),
        **extra,
    )
