"""pep8 shim for histoqc.MorphologyModule with pep484 type annotations"""
from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.MorphologyModule import removeSmallObjects as _removeSmallObjects
from histoqc.MorphologyModule import removeFatlikeTissue as _removeFatlikeTissue
from histoqc.MorphologyModule import fillSmallHoles as _fillSmallHoles

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable
    from .base_image import MaskStatisticsType

__all__ = [
    'remove_small_objects',
    'remove_fatlike_tissue',
    'fill_small_holes',
]


def remove_small_objects(
    pstate: PipelineCallable,
    *,
    min_size: int = 64,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> Optional[np.ndarray]:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _removeSmallObjects,
        min_size=min_size,
        **extra,
    )


def remove_fatlike_tissue(
    pstate: PipelineCallable,
    *,
    fat_cell_size: int = 64,
    kernel_size: int = 3,
    max_keep_size: int = 1000,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> Optional[np.ndarray]:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _removeFatlikeTissue,
        fat_cell_size=fat_cell_size,
        kernel_size=kernel_size,
        max_keep_size=max_keep_size,
        **extra,
    )


def fill_small_holes(
    pstate: PipelineCallable,
    *,
    min_size: int = 64,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> Optional[np.ndarray]:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _fillSmallHoles,
        min_size=min_size,
        **extra,
    )
