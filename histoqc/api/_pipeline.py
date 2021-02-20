import copy
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
from typing import TypedDict

import numpy as np

__all__ = [
    "PipelineCallable",
    "PipelineChain",
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
    updated: Dict[str, Any]
    added: Dict[str, Any]
    warnings: List[str]


class _BaseImageHelper(ChainMap):

    def __init__(self, i, c, s, image_thumb_getter: Callable[[str], np.ndarray]):
        u = self._reset_updates()
        super().__init__(u, i, c, s)
        self._print_list = {}
        self._image_thumb_getter = image_thumb_getter

    @staticmethod
    def _reset_updates(updates: Optional[dict] = None):
        u = {"warnings": []}
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
        return self._image_thumb_getter(dim)

    def collect_state(self, *, clear: bool = True) -> _PUpdate:
        """collect state changes"""
        updates, image_info, pconfig, pstate = self.maps

        _overlap = set(updates).intersection(image_info)
        if _overlap:
            raise RuntimeError(f"image_info key(s): {_overlap!r} updated")
        _overlap = set(updates).intersection(pconfig)
        if _overlap:
            raise RuntimeError(f"pipeline_config key(s): {_overlap!r} updated")

        # collect all info
        if not clear:
            updates = copy.copy(updates)
        warnings = updates.pop("warnings")

        updated_state = {k: updates.pop(k) for k in pstate if k in updates}
        added_state = {**updates}

        # reset updates
        self._reset_updates(updates)

        # return state
        return _PUpdate(updated_state, added_state, warnings)


class PipelineCallable(Protocol):

    def histoqc_call(self, func: Callable[[_BaseImageHelper, Dict[str, Any]], Any], **params) -> np.ndarray:
        ...


class PipelineState:

    def __init__(
        self,
        image_info: _ImageInfo,
        pipe_config: _PConfig,
        pipe_state: Optional[_PState] = None
    ):
        if pipe_state is None:
            pipe_state = self._init_state(image_info, pipe_config)
        self._image_info: _ImageInfo = image_info
        self._pconfig: _PConfig = pipe_config
        self._pstate: _PState = pipe_state
        self._warnings = []

    @property
    def warnings(self) -> List[str]:
        return self._warnings

    @property
    def mask(self) -> np.ndarray:
        return self._pstate["img_mask_use"]

    @contextmanager
    def base_image(self, *, update_state: bool = True, raise_if_warnings: bool = False):
        s = _BaseImageHelper(
            self._image_info, self._pconfig, self._pstate,
            image_thumb_getter=self.get_image_thumbnail
        )
        # yield the state
        yield s
        state_changes = s.collect_state()
        self._warnings.extend(state_changes.warnings)
        if raise_if_warnings and state_changes.warnings:
            w = state_changes.warnings
            raise RuntimeError(f"caught {len(w)} warnings: {w!r}")
        if update_state:
            self._pstate.update(state_changes.added)
            self._pstate.update(state_changes.updated)

    def get_image_thumbnail(self, dimension, *, image_info: Optional[_ImageInfo] = None) -> np.ndarray:
        ...

    @staticmethod
    def _init_state(self, image_info: _ImageInfo, pipe_config: _PConfig) -> _PState:
        dim = pipe_config["image_work_size"]
        _thumbnail_arr = self.get_image_thumbnail(dim, image_info=image_info)
        return _PState(
            img_mask_use=np.ones(_thumbnail_arr.shape[0:2], dtype=bool)
        )

    @classmethod
    def from_image(cls, image_fn: str, pipe_config: Optional[_PConfig] = None) -> 'PipelineCallable':
        ...

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
