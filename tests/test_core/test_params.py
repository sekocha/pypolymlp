"""Tests of params.py."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.params import PolymlpParams, set_common_params

cwd = Path(__file__).parent


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

    for p in params:
        assert p.type_full
        assert p.type_indices == [0]

    params.print_params()


def test_PolymlpParams3(regdata_mp_149):
    """Test params setters in PolymlpParams."""
    params_in, _ = regdata_mp_149
    params_single = copy.deepcopy(params_in.params)

    params = PolymlpParams()
    params.params = params_single
    assert len(params) == 1
    params.params = [params_single, params_single, params_single]
    assert len(params) == 3
    params.append(params_single)
    assert len(params) == 4


def test_PolymlpParams4(regdata_mp_149):
    """Test setters in PolymlpParams."""
    params_in, _ = regdata_mp_149
    params_single = copy.deepcopy(params_in.params)
    params = PolymlpParams([params_single, params_single])

    params.include_force = False
    assert not params.include_force
    for p in params:
        assert not p.include_force

    params.include_stress = False
    assert not params.include_stress
    for p in params:
        assert not p.include_stress

    params.dataset_type = "vasp"
    assert params.dataset_type == "vasp"
    for p in params:
        assert p.dataset_type == "vasp"

    params.temperature = 200
    assert params.temperature == 200
    for p in params:
        assert p.temperature == 200

    params.electron_property = "entropy"
    assert params.electron_property == "entropy"
    for p in params:
        assert p.electron_property == "entropy"

    params.element_swap = False
    assert not params.element_swap
    for p in params:
        assert not p.element_swap

    params.print_memory = True
    assert params.print_memory
    for p in params:
        assert p.print_memory

    params.regression_alpha = (-1, 0, 1)
    assert params.regression_alpha == (-1, 0, 1)
    for p in params:
        assert p.regression_alpha == (-1, 0, 1)

    params.alphas = (100, 50)
    assert params.alphas == (100, 50)
    for p in params:
        assert p.alphas == (100, 50)


def test_set_common_params(regdata_mp_149):
    """Test set_common_params."""
    params_in, _ = regdata_mp_149
    params_single = copy.deepcopy(params_in.params)
    multiple_params = [params_single, params_single]

    multiple_params[1].n_type = 2
    multiple_params[1].elements = ("Si", "Ge")
    multiple_params[1].atomic_energy = (0.1, 0.2)

    common_params = set_common_params(multiple_params)
    assert common_params.n_type == 2
    assert common_params.elements == ("Si", "Ge")
    assert common_params.atomic_energy == (0.1, 0.2)
