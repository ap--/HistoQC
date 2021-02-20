"""pep8 shim for histoqc.HistogramModule with pep484 type annotations"""
from typing import TYPE_CHECKING

from histoqc.HistogramModule import getHistogram as _getHistogram
from histoqc.HistogramModule import compareToTemplates as _compareToTemplates

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineState

__all__ = [
    'get_histogram',
    'compare_to_templates',
]


def get_histogram(
    pstate: PipelineState,
    *,
    limit_to_mask: bool = True,
    bins: int = 20,
) -> np.ndarray:
    return pstate.call(
        _getHistogram,
        limit_to_mask=str(limit_to_mask),
        bins=bins,
    )


def compare_to_templates(
    pstate: PipelineState,
    *,
    templates: str,
    limit_to_mask: bool = True,
    bins: int = 20,
) -> np.ndarray:
    return pstate.call(
        _compareToTemplates,
        templates=templates,
        limit_to_mask=str(limit_to_mask),
        bins=bins,
    )
