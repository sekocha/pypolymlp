"""Tests of elastic constant calculations."""

import copy
import os
from pathlib import Path

import numpy as np

from pypolymlp.calculator.compute_elastic import PolymlpElastic

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def _assert_elastic_pair_MgO(consts: np.ndarray):
    """Assert elastic constants from pair polymlp in MgO."""
    true1 = np.array(
        [
            [228.991191, 103.028144, 103.028144],
            [103.028144, 228.991191, 103.028144],
            [103.028144, 103.028144, 228.991191],
        ]
    )
    np.testing.assert_allclose(consts[:3, :3], true1, atol=1e-8)
    np.testing.assert_allclose(consts[:3, 3:], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, :3], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, 3:], np.eye(3) * 149.41388776, atol=1e-8)


def _assert_elastic_gtinv_MgO(consts: np.ndarray):
    """Assert elastic constants from pair polymlp in MgO."""
    true1 = np.array(
        [
            [288.21100485, 109.44248562, 109.44248562],
            [109.44248562, 288.21100485, 109.44248562],
            [109.44248562, 109.44248562, 288.21100485],
        ]
    )

    np.testing.assert_allclose(consts[:3, :3], true1, atol=1e-8)
    np.testing.assert_allclose(consts[:3, 3:], 0.0, atol=1e-8)
    np.testing.assert_allclose(consts[3:, :3], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        consts[3:, 3:], np.eye(3) * 149.55595257029483, atol=1e-8
    )


def test_pair_MgO(unitcell_pair_MgO):
    """Test elastic constants with pair polymlp in MgO."""
    unitcell1, pot = unitcell_pair_MgO
    unitcell = copy.deepcopy(unitcell1)
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    el = PolymlpElastic(
        unitcell=unitcell,
        unitcell_poscar=poscar,
        pot=pot,
        verbose=True,
    )
    el.run()
    _assert_elastic_pair_MgO(el.elastic_constants)

    el.write_elastic_constants("tmp.yaml")
    os.remove("tmp.yaml")


def test_gtinv_MgO(unitcell_gtinv_MgO):
    """Test elastic constants with polymlp in MgO."""
    unitcell1, pot = unitcell_gtinv_MgO
    unitcell = copy.deepcopy(unitcell1)
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    el = PolymlpElastic(
        unitcell=unitcell,
        unitcell_poscar=poscar,
        pot=pot,
        verbose=True,
    )
    el.run()
    _assert_elastic_gtinv_MgO(el.elastic_constants)
