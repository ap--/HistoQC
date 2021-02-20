"""pep8 shim for histoqc.ClassificationModule with pep484 type annotations"""
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.ClassificationModule import pixelWise as _pixelWise
from histoqc.ClassificationModule import byExampleWithFeatures as _byExampleWithFeatures

if TYPE_CHECKING:
    from threading import Lock
    import numpy as np
    from ._pipeline import PipelineState
    from .base_image import MaskStatisticsType

__all__ = [
    'pixel_wise',
    'by_example_with_features',
]


def pixel_wise(
    pstate: PipelineState,
    *,
    tsv_file: str,
    name: str = "classTask",
    threshold: float = 0.01,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> np.ndarray:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.call(
        _pixelWise,
        name=name,
        tsv_file=tsv_file,
        threshold=threshold,
        **extra,
    )


def by_example_with_features(
    pstate: PipelineState,
    *,
    examples: str,
    features: str,
    lock: Lock,
    shared_dict: dict,
    name: str = "classTask",
    threshold: float = 0.5,
    area_threshold: int = 5,
    dilate_kernel_size: int = 0,
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> np.ndarray:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.call(
        _byExampleWithFeatures,
        name=name,
        threshold=threshold,
        examples=examples,
        features=features,
        lock=lock,
        shared_dict=shared_dict,
        area_threshold=area_threshold,
        dilate_kernel_size=dilate_kernel_size,
        **extra,
    )
