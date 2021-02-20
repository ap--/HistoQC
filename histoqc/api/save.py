"""pep8 shim for histoqc.SaveModule with pep484 type annotations"""
from typing import TYPE_CHECKING

from histoqc.SaveModule import saveFinalMask as _saveFinalMask
from histoqc.SaveModule import saveThumbnails as _saveThumbnails

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable

__all__ = [
    'save_final_mask',
    'save_thumbnails',
]


def save_final_mask(
    pstate: PipelineCallable,
    *,
    use_mask: bool = True,
) -> np.ndarray:
    return pstate.histoqc_call(
        _saveFinalMask,
        use_mask=str(use_mask),
    )


def save_thumbnails(
    pstate: PipelineCallable,
    *,
    image_work_size: str = "1.25x",
    small_dim: int = 500,
) -> np.ndarray:
    return pstate.histoqc_call(
        _saveThumbnails,
        image_work_size=image_work_size,
        small_dim=small_dim,
    )
