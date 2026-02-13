"""Tests of force constant calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_fc_AlN():
    """Test FC calculation."""
    poscar = path_file + "poscars/POSCAR.WZ.AlN"
    pot = path_file + "mlps/polymlp.lammps.gtinv.AlN"

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.load_poscars(poscar)

    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization()
    polymlp.init_fc(supercell_matrix=(3, 3, 2), cutoff=3.0)
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
    assert fc3.shape == (4, 72, 72, 3, 3, 3)
    assert np.sum(fc2) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(fc3) == pytest.approx(0.0, abs=1e-6)


def test_fc_MgO():
    """Test FC calculation."""
    poscar = path_file + "poscars/POSCAR.RS.MgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.load_poscars(poscar)

    polymlp.init_geometry_optimization(
        with_sym=True,
        relax_cell=False,
        relax_volume=False,
        relax_positions=True,
    )
    polymlp.run_geometry_optimization()
    polymlp.init_fc(supercell_matrix=(2, 2, 2), cutoff=4.0)
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
    assert fc3.shape == (2, 64, 64, 3, 3, 3)
    assert np.sum(fc2) == pytest.approx(0.0, abs=1e-6)
    assert np.sum(fc3) == pytest.approx(0.0, abs=1e-6)
