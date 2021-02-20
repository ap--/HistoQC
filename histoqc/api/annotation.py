"""pep8 shim for histoqc.AnnotationModule with pep484 type annotations"""
from __future__ import annotations
from typing import Optional
from typing import TYPE_CHECKING

from histoqc.AnnotationModule import xmlMask as _xmlMask
from histoqc.AnnotationModule import geoJSONMask as _geoJSONMask

if TYPE_CHECKING:
    import numpy as np
    from ._pipeline import PipelineCallable

__all__ = ['xml_mask', 'geojson_mask']


def xml_mask(
    pstate: PipelineCallable,
    *,
    xml_filepath: Optional[str] = None,
    xml_suffix: str = "",
) -> Optional[np.ndarray]:
    return pstate.histoqc_call(_xmlMask, xml_filepath=xml_filepath, xml_suffix=xml_suffix)


def geojson_mask(
    pstate: PipelineCallable,
    *,
    geojson_filepath: Optional[str] = None,
    geojson_suffix: str = ""
) -> Optional[np.ndarray]:
    return pstate.histoqc_call(
        _geoJSONMask,
        geojson_filepath=geojson_filepath,
        geojson_suffix=geojson_suffix
    )
