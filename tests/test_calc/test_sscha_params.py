"""Tests of SSCHA parameter class."""

import numpy as np
import pytest

from pypolymlp.calculator.sscha.sscha_params import SSCHAParams


def test_params1(unitcell_mlp_Al):
    """Test SSCHAParams."""
    unitcell, pot = unitcell_mlp_Al
    size = (2, 2, 2)
    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temp_min=300,
        temp_max=500,
        temp_step=100,
    )
    temperatures_true = [500, 400, 300]
    np.testing.assert_equal(params.temperatures, temperatures_true)

    assert params.n_unitcells == 8
    assert params.n_atom == 32
    assert params.supercell_matrix.shape == (3, 3)
    assert params.tol == pytest.approx(0.005)
    assert params.max_iter == 50
    assert params.mixing == pytest.approx(0.5)
    assert params.init_fc_algorithm == "harmonic"
    assert params.init_fc_file is None

    params.set_n_samples_from_basis(n_basis=1000)
    assert params.n_samples_init == 79550
    assert params.n_samples_final == 79550 * 3
    params.print_unitcell()
    params.print_params()

    # Test setters
    params.supercell = unitcell
    params.pot = pot
    params.temperatures = temperatures_true
    np.testing.assert_equal(params.temperatures, temperatures_true)

    params.mesh = (10, 10, 10)
    params.nac_params = dict()
    params.cutoff_radius = 6.0


def test_params_temperatures(unitcell_mlp_Al):
    """Test SSCHAParams."""
    unitcell, pot = unitcell_mlp_Al
    size = (2, 2, 2)
    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temp=300,
    )
    true = [300]
    np.testing.assert_equal(params.temperatures, [300])

    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temp_min=300,
        temp_max=500,
        temp_step=100,
        ascending_temp=True,
    )
    true = [300, 400, 500]
    np.testing.assert_equal(params.temperatures, true)

    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temperatures=true,
    )
    np.testing.assert_equal(params.temperatures, true)

    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temp_min=300,
        temp_max=500,
        n_temp=5,
        ascending_temp=True,
    )
    true = [300, 329, 400, 471, 500]
    np.testing.assert_equal(params.temperatures, true)
