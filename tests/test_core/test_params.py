"""Tests of params.py."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.params import PolymlpParams

cwd = Path(__file__).parent

# TODO: unique_type, setters


def _assert_properties(params):
    """Assert properties."""
    assert params.n_type == 1
    assert tuple(params.elements) == ("Si",)
    assert tuple(params.element_order) == ("Si",)
    assert params.atomic_energy == (0.0,)
    assert params.include_force
    assert not params.include_stress
    assert params.dataset_type == "phono3py"
    assert params.temperature == 300
    assert params.electron_property == "free_energy"
    assert not params.element_swap
    assert not params.print_memory
    np.testing.assert_allclose(params.regression_alpha, [-3, -2, -1, 0, 1])
    np.testing.assert_allclose(params.alphas, [1e-3, 1e-2, 1e-1, 1e0, 1e1])


def test_PolymlpParams1(regdata_mp_149):
    """Test PolymlpParams."""
    params_in, _ = regdata_mp_149
    params_single = copy.deepcopy(params_in.params)
    params = PolymlpParams(params_single)

    common = params._common_params
    assert isinstance(common, PolymlpParamsSingle)
    assert common.n_type == 1
    assert tuple(common.elements) == ("Si",)
    assert tuple(common.element_order) == ("Si",)

    for p in params:
        assert isinstance(p, PolymlpParamsSingle)
    assert isinstance(params[0], PolymlpParamsSingle)
    assert len(params) == 1

    assert isinstance(params.params, PolymlpParamsSingle)
    _assert_properties(params)
    assert not params.is_hybrid
    assert isinstance(params.as_dict(), dict)
    params.print_params()


def test_PolymlpParams2(regdata_mp_149):
    """Test PolymlpParams for hybrid model."""
    params_in, _ = regdata_mp_149
    params_single = copy.deepcopy(params_in.params)
    params = PolymlpParams(params_single)
    params = params.as_hybrid_model()

    common = params._common_params
    assert isinstance(common, PolymlpParamsSingle)
    assert common.n_type == 1
    assert tuple(common.elements) == ("Si",)
    assert tuple(common.element_order) == ("Si",)

    for p in params:
        assert isinstance(p, PolymlpParamsSingle)
    assert isinstance(params[0], PolymlpParamsSingle)
    assert isinstance(params[1], PolymlpParamsSingle)
    assert len(params) == 2

    assert isinstance(params.params, list)
    _assert_properties(params)
    assert params.is_hybrid

    for d in params.as_dict():
        assert isinstance(d, dict)

    params.print_params()

    params.append(params_single)
    assert len(params) == 3
