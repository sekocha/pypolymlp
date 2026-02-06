"""Tests of API for SSCHA functions."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from test_sscha_api_sscha import _assert_Al

from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"

unitcell = Poscar(poscar).structure
size = (2, 2, 2)


def test_sscha_core():
    """Test run_sscha."""
    sscha_params = SSCHAParams(unitcell, size, pot=pot, temp=700, tol=0.003)
    sscha = SSCHACore(sscha_params, pot=pot, verbose=True)

    assert sscha._n_atom == 32
    assert sscha._n_unitcells == 8
    assert sscha_params._n_samples_init == 3138
    assert sscha_params._n_samples_final == 3138 * 3
    assert sscha._n_coeffs == 11
    assert sscha._ph_real is not None
    assert sscha._ph_recip is not None
    assert sscha_params.supercell is not None

    sscha_params.init_fc_algorithm = "const"
    sscha.set_initial_force_constants()
    assert sscha.force_constants.shape == (32, 32, 3, 3)

    sscha_params.init_fc_algorithm = "random"
    sscha.force_constants = None
    sscha.set_initial_force_constants()
    assert sscha.force_constants.shape == (32, 32, 3, 3)

    sscha_params.init_fc_algorithm = "harmonic"
    sscha.force_constants = None
    sscha.set_initial_force_constants()
    assert sscha.force_constants.shape == (32, 32, 3, 3)

    freq = sscha.run_frequencies(qmesh=(5, 5, 5))
    assert np.average(freq) == pytest.approx(6.033117718600683)

    data = sscha._compute_sscha_properties(temp=700)
    assert data is not None

    rec_fc2 = sscha._recover_fc2(np.random.random(11))
    assert rec_fc2.shape == (32, 32, 3, 3)

    score = sscha._convergence_score(np.array([0.1, 0.2]), np.array([0.2, 0.1]))
    assert score == pytest.approx(0.6324555320336759)

    sscha._single_iter(temp=700, n_samples=10)
    assert sscha._data_current.delta < 0.5
    assert sscha._data_current.anharmonic_free_energy is not None
    assert sscha._data_current.free_energy is not None
    sscha._final_iter(temp=700, n_samples=10)

    assert not sscha._is_imaginary()

    sscha.precondition(temp=700, n_samples=10, max_iter=1)
    sscha.run(temp=700)
    sscha._print_progress()

    sscha._write_dos(filename="tmp/total_dos.dat", write_pdos=True)
    sscha._print_final_results()
    sscha.save_results(path="tmp")
    shutil.rmtree("tmp")

    _assert_Al(sscha)

    assert sscha.sscha_params is not None
    assert sscha.properties is not None
    assert len(sscha.logs) > 2
    assert sscha.n_fc_basis == 11
