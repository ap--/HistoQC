from __future__ import annotations
import copy
import os
import sys
import types
import warnings
from collections import ChainMap
from contextlib import contextmanager
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import cast

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import numpy as np

# imports for alternative PipelineChain call interface
from . import _functions as _functions
from ..BaseImage import BaseImage

__all__ = [
    "PipelineCallable",
    "PipelineChain",
    "PipelineState",
]


class _ImageInfo(TypedDict):
    filename: str
    dir: str
    image_base_size: Tuple[int, int]


class _PConfig(TypedDict):
    mask_statistics: str
    outdir: str
    image_work_size: str  # string: 1.25x or 1.23 or 0.123 or 2..100


class _PState(TypedDict):
    img_mask_use: np.ndarray


class _PUpdate(NamedTuple):
    updated: _PState
    added: Dict[str, Any]
    warnings: List[str]


class _BaseImageHelper(ChainMap):

    def __init__(self, base_image: BaseImage):
        self._base_image = base_image
        u = self._reset_updates()
        super().__init__(u, base_image)
        self._print_list: Dict[str, str] = {}

    @staticmethod
    def _reset_updates(updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        u: Dict[str, Any] = {"warnings": []}
        if updates is None:
            return u
        else:
            updates.clear()
            updates.update(**u)
            return updates

    # noinspection PyPep8Naming
    def addToPrintList(self, name: str, value: str):
        self._print_list[name] = value

    # noinspection PyPep8Naming
    def getImgThumb(self, dim):
        return self._base_image.getImgThumb(dim)

    def collect_state(self, *, clear: bool = True) -> _PUpdate:
        """collect state changes"""
        updates = cast(Dict[str, Any], self.maps[0])

        # collect all info
        if not clear:
            updates = copy.copy(updates)
        _warnings = updates.pop("warnings")

        updated_state = {k: updates.pop(k) for k in self._base_image if k in updates}
        added_state = {**updates}

        # reset updates
        self._reset_updates(updates)

        # return state
        return _PUpdate(updated_state, added_state, _warnings)


class PipelineCallable(Protocol):

    def histoqc_call(self, func, **params) -> Optional[np.ndarray]:
        ...


class PipelineState:

    # TODO: make calling interface exlicit
    def __init__(
        self,
        fname,
        fname_outdir,
        params
    ):
        fname = os.fspath(fname)
        fname_outdir = os.fspath(fname_outdir)
        self._base_image = BaseImage(fname, fname_outdir, params)

    @property
    def warnings(self) -> List[str]:
        return self._base_image["warnings"]

    @property
    def mask(self) -> np.ndarray:
        return self._base_image["img_mask_use"]

    @contextmanager
    def base_image(self, *, update_state: bool = True, raise_if_warnings: bool = False):
        s = _BaseImageHelper(self._base_image)
        # yield the state
        yield s
        state_changes = s.collect_state()
        self.warnings.extend(state_changes.warnings)
        if raise_if_warnings and state_changes.warnings:
            w = state_changes.warnings
            raise RuntimeError(f"caught {len(w)} warnings: {w!r}")
        if update_state:
            if state_changes.added:
                warnings.warn(f"added keys: {state_changes.added!r} to state")
            self._base_image.update(
                **state_changes.added,
                **state_changes.updated,
            )  # type: ignore

    def histoqc_call(self, func, **params) -> np.ndarray:
        with self.base_image() as s:
            _ = func(s, params)
        return self.mask


class PipelineChain:

    def __init__(self):
        self._steps = []

    def histoqc_call(self, func, **params) -> None:
        self._steps.append((func, params))
        return None

    def run(self, pstate: PipelineState) -> np.ndarray:
        with pstate.base_image() as s:
            for func, params in self._steps:
                _ = func(s, params)
        return pstate.mask

    def __dir__(self):
        # update __dir__ with functions in submodules
        _dir = list(super().__dir__())
        _dir.extend(
            name for name in _functions.__all__
            if isinstance(getattr(_functions, name), types.FunctionType)
        )
        return _dir

    def __getattr__(self, item):
        # allow chain.my_function_name call style
        obj = getattr(_functions, item)
        if isinstance(obj, types.FunctionType):
            return obj.__get__(self, self.__class__)
        else:
            raise AttributeError(item)
