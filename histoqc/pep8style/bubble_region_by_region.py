"""pep8 shim for histoqc.BubbleRegionByRegion with pep484 type annotations"""
from typing import Optional

import numpy as np

from histoqc.pep8style._pipeline import PipelineState
from histoqc.pep8style.base_image import MaskStatisticsType
from histoqc.BubbleRegionByRegion import roiWise as _roiWise
from histoqc.BubbleRegionByRegion import detectSmoothness as _detectSmoothness

__all__ = [
    'roi_wise',
    'detect_smoothness',
]


def roi_wise(
    pstate: PipelineState,
    *,
    name: str = "classTask",
    level: int = 1,
    win_size: int = 2048,
    area_threshold: Optional[int] = None,
) -> np.ndarray:
    if area_threshold is None:
        area_threshold = ""
    return pstate.call(
        _roiWise, name=name, level=level, win_size=win_size, area_threshold=area_threshold
    )


def detect_smoothness(
    pstate: PipelineState,
    *,
    threshold: float = 0.01,
    kernel_size: int = 10,
    min_object_size: int = 100,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> np.ndarray:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.call(
        _detectSmoothness,
        threshold=threshold,
        kernel_size=kernel_size,
        min_object_size=min_object_size,
        **extra,
    )
