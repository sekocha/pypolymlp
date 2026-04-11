"""Tests of ti_utils."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.thermodynamics.ti_utils import (
    DataTI,
    _check_melting,
    _extrapolate_data,
    _get_energy,
    _get_free_energy,
    _get_properties,
    _is_success,
    integrate,
    load_ti_yaml,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"

file = path_file + "ti/00024/1000.0/polymlp_ti.yaml"
properties, properties_ext = load_ti_yaml(filename=file)


def test_load_ti_yaml():
    """Test load_ti_yaml."""
    assert isinstance(properties, DataTI)
    assert isinstance(properties_ext, DataTI)
    assert properties.volume == pytest.approx(12.186720948018055)
    assert properties.temperature == 1000
    assert properties.free_energy == pytest.approx(0.0006251902850217512)
    assert properties.energy == pytest.approx(-0.001229930604749909)
    assert properties.entropy == pytest.approx(-1.85512088977166e-06)


def test_property_functions():
    """Test local functions in ti_utils."""
    log = properties.log
    temp = properties.temperature
    n_atom = 108
    F, E, S = _get_properties(log, temp, n_atom, extrapolation=False)
    assert F == pytest.approx(0.0006251902850217512)
    assert E == pytest.approx(-0.001229930604749909)
    assert S == pytest.approx(-1.85512088977166e-06)

    F = _get_free_energy(log, extrapolation=False, method="trapezoid")
    assert F / n_atom == pytest.approx(0.00028355700375388813)
    F = _get_free_energy(log, extrapolation=False, method="simpson")
    assert F / n_atom == pytest.approx(0.0006251902850217512)
    E = _get_energy(log, extrapolation=False)
    assert E / n_atom == pytest.approx(-0.001229930604749909)

    F, E, S = _get_properties(log, temp, n_atom, extrapolation=True)
    assert F == pytest.approx(-0.00022588635710226365)
    assert E == pytest.approx(-0.0013240537335495563)
    assert S == pytest.approx(-1.0981673764472929e-06)

    F = _get_free_energy(log, extrapolation=True, method="trapezoid")
    assert F / n_atom == pytest.approx(-0.0005464876501016688)
    F = _get_free_energy(log, extrapolation=True, method="simpson")
    assert F / n_atom == pytest.approx(-0.00022588635710226365)
    E = _get_energy(log, extrapolation=True)
    assert E / n_atom == pytest.approx(-0.0013240537335495563)


def test_local_functions():
    """Test local functions in ti_utils."""
    assert _is_success(0.1)
    is_melt = _check_melting([0.1, 0.13, 0.16, 0.2, 0.22, 0.25, 0.3])
    np.testing.assert_equal(is_melt, [0, 0, 0, 0, 1, 1, 1])

    data = np.array([[0, 2.0], [0.5, 3.0], [1.0, 4.0]])
    integ = integrate(data, method="trapezoid")
    assert integ == pytest.approx(3.0)

    integ = integrate(data, method="simpson")
    assert integ == pytest.approx(3.0)

    log = properties.log
    data_de = np.array([[float(l["alpha"]), float(l["delta_e"])] for l in log])
    val1 = _extrapolate_data(data_de, max_order=4, threshold=0.7)
    assert val1 == pytest.approx(-4.722074464741127)
    val1 = _extrapolate_data(data_de, max_order=6, threshold=0.7)
    assert val1 == pytest.approx(-4.722074464741127)
    val1 = _extrapolate_data(data_de, max_order=4, threshold=0.5)
    assert val1 == pytest.approx(-4.7046040517376895)
    val1 = _extrapolate_data(data_de, max_order=6, threshold=0.5)
    assert val1 == pytest.approx(-4.7046040517376895)
