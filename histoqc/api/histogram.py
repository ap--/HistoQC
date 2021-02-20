"""pep8 shim for histoqc.HistogramModule with pep484 type annotations"""
from __future__ import annotations
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.HistogramModule import getHistogram as _getHistogram
from histoqc.HistogramModule import compareToTemplates as _compareToTemplates

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable

__all__ = [
    'get_histogram',
    'compare_to_templates',
]


def get_histogram(
    pstate: PipelineCallable,
    *,
    limit_to_mask: bool = True,
    bins: int = 20,
) -> Optional[np.ndarray]:
    return pstate.histoqc_call(
        _getHistogram,
        limit_to_mask=str(limit_to_mask),
        bins=bins,
    )


def compare_to_templates(
    pstate: PipelineCallable,
    *,
    templates: List[str],
    limit_to_mask: bool = True,
    bins: int = 20,
) -> Optional[np.ndarray]:
    return pstate.histoqc_call(
        _compareToTemplates,
        templates="\n".join(templates),
        limit_to_mask=str(limit_to_mask),
        bins=bins,
    )
