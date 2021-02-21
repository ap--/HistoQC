import types

import numpy as np
import pytest

import histoqc.api as qc
from histoqc.data import package_resource_copytree


@pytest.fixture(scope="function")
def pipeline_state(svs_small, tmp_path):
    yield qc.PipelineState(fname=svs_small, fname_outdir=tmp_path, params={})


def test_pipeline_state(svs_small, tmp_path):
    s = qc.PipelineState(fname=svs_small, fname_outdir=tmp_path, params={})
    assert isinstance(s.mask, np.ndarray)
    assert np.all(s.mask), "initial mask is restricted?"


FUNCTIONS = [
    obj
    for obj in (getattr(qc, obj_name) for obj_name in qc.__all__)
    if isinstance(obj, types.FunctionType)
]


def _idfn(val):
    if isinstance(val, types.BuiltinFunctionType):
        return val.__qualname__


@pytest.mark.parametrize(
    "histoqc_func",
    argvalues=FUNCTIONS,
    ids=_idfn,
)
def test_calling_with_defaults(pipeline_state, histoqc_func):
    try:
        mask = histoqc_func(pipeline_state)
    except TypeError as err:
        if "required keyword-only" in str(err):
            pytest.skip("no defaults for required kwargs")
        raise
    assert mask is not None


# --- test the functions that require kwargs --------------------------

@pytest.fixture(scope='module')
def pkg_data(tmp_path_factory):
    base_folder = tmp_path_factory.mktemp('histoqc_pkg_data')
    for rsrc in {'models', 'pen', 'templates'}:
        package_resource_copytree('histoqc.data', rsrc, base_folder)
    yield base_folder


def test_classification_pixel_wise(pipeline_state, pkg_data):
    tsv_file = pkg_data.joinpath("models", "pen_markings_he", "he.tsv")
    mask = qc.pixel_wise(pipeline_state, tsv_file=str(tsv_file))
    assert mask is not None


def test_classification_by_example_with_features(pipeline_state, pkg_data):
    from threading import Lock

    _pth = pkg_data.joinpath("pen", "1k_version")
    examples = ":".join(
        str(_pth.joinpath(fn)) for fn in ["pen_green.png", "pen_green_mask.png"]
    )
    features = "\n".join(["frangi", "laplace", "rgb"])

    mask = qc.by_example_with_features(
        pipeline_state,
        examples=examples,
        features=features,
        lock=Lock(),
        shared_dict={},
    )
    assert mask is not None


def test_deconvolution_separate_stains(pipeline_state):
    mask = qc.separate_stains(pipeline_state, stain="hdx_from_rgb")
    assert mask is not None


def test_histogram_compare_to_templates(pipeline_state, pkg_data):
    _pth = pkg_data.joinpath("templates")
    files = ["template1.png", "template2.png", "template3.png", "template4.png"]
    templates = [str(_pth.joinpath(fn)) for fn in files]

    mask = qc.compare_to_templates(pipeline_state, templates=templates)
    assert mask is not None


def test_pipeline_chain(pipeline_state):
    c = qc.PipelineChain()
    qc.get_contrast(c)
    qc.get_histogram(c)
    assert c.run(pipeline_state) is not None
