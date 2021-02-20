"""pep8 shim for histoqc.BlurDetectionModule with pep484 type annotations"""
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.BlurDetectionModule import identifyBlurryRegions as _identifyBlurryRegions

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable
    from .base_image import MaskStatisticsType

__all__ = ['identify_blurry_regions']


def identify_blurry_regions(
    pstate: PipelineCallable,
    *,
    blur_radius: int = 7,
    blur_threshold: float = 0.1,
    image_work_size: str = "2.5x",
    mask_statistics: Optional[MaskStatisticsType] = None,
) -> np.ndarray:
    extra = {}
    if mask_statistics is not None:
        extra["mask_statistics"] = mask_statistics
    return pstate.histoqc_call(
        _identifyBlurryRegions,
        blur_radius=blur_radius,
        blur_threshold=blur_threshold,
        image_work_size=image_work_size,
        **extra,
    )
