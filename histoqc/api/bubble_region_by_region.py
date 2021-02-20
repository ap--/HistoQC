"""pep8 shim for histoqc.BubbleRegionByRegion with pep484 type annotations"""
from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.BubbleRegionByRegion import roiWise as _roiWise
from histoqc.BubbleRegionByRegion import detectSmoothness as _detectSmoothness

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable
    from .base_image import MaskStatisticsType

__all__ = [
    'roi_wise',
    'detect_smoothness',
]


def roi_wise(
    pstate: PipelineCallable,
    *,
    name: str = "classTask",
    level: int = 1,
    win_size: int = 2048,
    area_threshold: Optional[int] = None,
) -> Optional[np.ndarray]:
    return pstate.histoqc_call(
        _roiWise,
        name=name,
        level=level,
        win_size=win_size,
        area_threshold=area_threshold if area_threshold is not None else ""  # used as default in _roiWise
    )


def detect_smoothness(
    pstate: PipelineCallable,
    *,
    threshold: float = 0.01,
    kernel_size: int = 10,
    min_object_size: int = 100,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> Optional[np.ndarray]:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _detectSmoothness,
        threshold=threshold,
        kernel_size=kernel_size,
        min_object_size=min_object_size,
        **extra,
    )
