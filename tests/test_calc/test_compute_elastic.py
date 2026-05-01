"""Tests of elastic constant calculations."""

import copy
import os
import shutil
from pathlib import Path

import numpy as np

from pypolymlp.calculator.compute_elastic import PolymlpElastic
from pypolymlp.calculator.sscha.api_properties import PropertiesSSCHA
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def _assert_elastic_pair_MgO(consts: np.ndarray):
    """Assert elastic constants from pair polymlp in MgO."""
    np.set_printoptions(suppress=True)
    print(consts)
    true1 = np.array(
        [
            [231.314072, 103.989877, 103.989877],
            [103.989877, 231.314072, 103.989877],
            [103.989877, 103.989877, 231.314072],
        ]
    )
    np.testing.assert_allclose(consts[:3, :3], true1, atol=1e-8)
    np.testing.assert_allclose(consts[:3, 3:], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, :3], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, 3:], np.eye(3) * 143.13633497, atol=1e-8)


def _assert_elastic_gtinv_MgO(consts: np.ndarray):
    """Assert elastic constants from pair polymlp in MgO."""
    print(consts)
    true1 = np.array(
        [
            [288.38350811, 109.43998159, 109.43998159],
            [109.43998159, 288.38350811, 109.43998159],
            [109.43998159, 109.43998159, 288.38350811],
        ]
    )
    np.testing.assert_allclose(consts[:3, :3], true1, atol=1e-8)
    np.testing.assert_allclose(consts[:3, 3:], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, :3], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, 3:], np.eye(3) * 154.74538473, atol=1e-6)


def test_pair_MgO(unitcell_pair_MgO):
    """Test elastic constants with pair polymlp in MgO."""
    unitcell1, pot, prop = unitcell_pair_MgO
    unitcell = copy.deepcopy(unitcell1)
    el = PolymlpElastic(unitcell=unitcell, properties=prop, verbose=True)
    el.run()
    _assert_elastic_pair_MgO(el.elastic_constants)

    el.write_elastic_constants("tmp.yaml")
    os.remove("tmp.yaml")


def test_gtinv_MgO(unitcell_gtinv_MgO):
    """Test elastic constants with polymlp in MgO."""
    unitcell1, pot, prop = unitcell_gtinv_MgO
    unitcell = copy.deepcopy(unitcell1)
    el = PolymlpElastic(unitcell=unitcell, properties=prop, verbose=True)
    el.run()
    _assert_elastic_gtinv_MgO(el.elastic_constants)


def test_pair_MgO_sscha(unitcell_pair_MgO):
    """Test SSCHA elastic constants with pair polymlp in MgO."""
    unitcell1, pot, prop = unitcell_pair_MgO
    unitcell = copy.deepcopy(unitcell1)

    sscha_params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=np.eye(3),
        pot=pot,
        temp=300,
        tol=0.1,
        mixing=0.99,
    )
    prop_sscha = PropertiesSSCHA(sscha_params, prop, verbose=True)

    el = PolymlpElastic(unitcell=unitcell, properties=prop_sscha, verbose=True)
    el.run(n_samples=3, eps=0.01)
    el.run_adiabatic(n_samples=3, eps=60)
    el.write_elastic_constants(filename="tmp_sscha.yaml")
    os.remove("tmp_sscha.yaml")
    shutil.rmtree("sscha")
