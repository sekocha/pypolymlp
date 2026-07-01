"""Tests of calculations using lammps API."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent


path = str(cwd) + "/files/Ti-Al/"
poscar = path + "POSCAR-Ti"


def test_elastic(property_mlp_Ti_Al):
    """Test elastic constants."""
    polymlp = PypolymlpCalc(properties=property_mlp_Ti_Al, verbose=True)
    polymlp.run_elastic_constants(poscar=poscar)
    assert np.sum(polymlp.elastic_constants) == pytest.approx(1418.3332019314498)


def test_eos(property_mlp_Ti_Al):
    """Test EOS."""
    polymlp = PypolymlpCalc(properties=property_mlp_Ti_Al, verbose=True)
    polymlp.load_poscars(poscar)
    polymlp.run_eos(
        eps_min=0.7,
        eps_max=2.0,
        eps_step=0.03,
        fine_grid=True,
        eos_fit=True,
    )
    e0, v0, b0 = polymlp.eos_fit_data
    assert e0 == pytest.approx(-10.430119364166558)
    assert v0 == pytest.approx(34.18362432994893)
    assert b0 == pytest.approx(123.89753096431376)


def test_geometry_opt(property_mlp_Ti_Al):
    """Test geometry optimization."""
    polymlp = PypolymlpCalc(properties=property_mlp_Ti_Al, verbose=True)
    polymlp.load_poscars(poscar)
    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization(gtol=1e-5, method="BFGS")
    e0, n_iter, success = polymlp.go_data

    assert e0 == pytest.approx(-10.428506392939765, rel=1e-6)
    np.testing.assert_allclose(polymlp._go.residual_forces[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(polymlp._go.residual_forces[1], 0.0, atol=1e-5)

    e, f, s = polymlp.eval(polymlp.converged_structure)
    assert e[0] == pytest.approx(-10.428506392939765, rel=1e-6)
    np.testing.assert_allclose(f, 0.0, atol=1e-4)
    np.testing.assert_allclose(s, 0.0, atol=1e-4)


def test_fc(property_mlp_Ti_Al):
    """Test force constant calculations."""
    polymlp = PypolymlpCalc(properties=property_mlp_Ti_Al, verbose=True)
    polymlp.load_poscars(poscar)
    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization()
    polymlp.init_fc(supercell_matrix=(2, 2, 2), cutoff={3: 3.0})
    polymlp.run_fc(
        n_samples=100,
        distance=0.005,
        is_plusminus=False,
        orders=(2, 3),
        batch_size=100,
        is_compact_fc=True,
        use_mkl=False,
    )
    fc2 = polymlp._fc.fc2
    fc3 = polymlp._fc.fc3
    assert fc3.shape == (1, 16, 16, 3, 3, 3)
    assert np.sum(fc2) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(fc3) == pytest.approx(0.0, abs=1e-6)


def test_phonon(property_mlp_Ti_Al):
    """Test phonon calculations."""
    polymlp = PypolymlpCalc(properties=property_mlp_Ti_Al, verbose=True)
    polymlp.load_poscars(poscars=poscar)
    polymlp.init_phonon(supercell_matrix=(2, 2, 2))
    polymlp.run_phonon()

    ph = polymlp._phonon
    assert ph.total_dos.shape == (201, 2)
    assert ph.is_imaginary


def test_phonon_qha(property_mlp_Ti_Al):
    """Test QHA calculations."""
    polymlp = PypolymlpCalc(properties=property_mlp_Ti_Al, verbose=True)
    polymlp.load_poscars(poscars=poscar)
    polymlp.run_qha(supercell_matrix=(1, 1, 1))

    qha = polymlp._qha
    assert len(qha.temperatures) == 101
    assert len(qha.bulk_modulus_temperature) == 100
    assert len(qha.thermal_expansion) == 100
