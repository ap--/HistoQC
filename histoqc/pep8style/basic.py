"""pep8 shim for histoqc.BasicModule with pep484 type annotations"""
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.BasicModule import getBasicStats as _getBasicStats
from histoqc.BasicModule import finalComputations as _finalComputations
from histoqc.BasicModule import finalProcessingSpur as _finalProcessingSpur
from histoqc.BasicModule import finalProcessingArea as _finalProcessingArea

if TYPE_CHECKING:
    import numpy as np
    from histoqc.pep8style._pipeline import PipelineState
    from histoqc.pep8style.base_image import MaskStatisticsType

__all__ = [
    'get_basic_stats',
    'final_computations',
    'final_processing_spur',
    'final_processing_area',
]


def get_basic_stats(pstate: PipelineState) -> np.ndarray:
    return pstate.call(_getBasicStats)


def final_computations(pstate: PipelineState) -> np.ndarray:
    return pstate.call(_finalComputations)


def final_processing_spur(
    pstate: PipelineState,
    *,
    disk_radius: int = 25,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> np.ndarray:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.call(_finalProcessingSpur, disk_radius=disk_radius, **extra)


def final_processing_area(
    pstate: PipelineState,
    *,
    area_threshold: int = 1000,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> np.ndarray:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.call(_finalProcessingArea, area_threshold=area_threshold, **extra)
