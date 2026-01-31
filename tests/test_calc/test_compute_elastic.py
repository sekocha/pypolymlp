"""Tests of elastic constant calculations."""

import os
from pathlib import Path

import numpy as np

from pypolymlp.calculator.compute_elastic import PolymlpElastic
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_pair_MgO():
    """Test elastic constants with pair polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    unitcell = Poscar(poscar).structure

    el = PolymlpElastic(
        unitcell=unitcell,
        unitcell_poscar=poscar,
        pot=pot,
        verbose=True,
    )
    el.run()
    const = el.elastic_constants

    true1 = np.array(
        [
            [228.991191, 103.028144, 103.028144],
            [103.028144, 228.991191, 103.028144],
            [103.028144, 103.028144, 228.991191],
        ]
    )
    np.testing.assert_allclose(const[:3, :3], true1, atol=1e-8)
    np.testing.assert_allclose(const[:3, 3:], 0.0, atol=1e-8)
    np.testing.assert_allclose(const[3:, :3], 0.0, atol=1e-8)
    np.testing.assert_allclose(const[3:, 3:], np.eye(3) * 149.41388776, atol=1e-8)

    el.write_elastic_constants("tmp.yaml")
    os.remove("tmp.yaml")


def test_gtinv_MgO():
    """Test elastic constants with polymlp in MgO."""
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    unitcell = Poscar(poscar).structure

    el = PolymlpElastic(
        unitcell=unitcell,
        unitcell_poscar=poscar,
        pot=pot,
        verbose=True,
    )
    el.run()
    const = el.elastic_constants

    true1 = np.array(
        [
            [288.21100485, 109.44248562, 109.44248562],
            [109.44248562, 288.21100485, 109.44248562],
            [109.44248562, 109.44248562, 288.21100485],
        ]
    )

    np.testing.assert_allclose(const[:3, :3], true1, atol=1e-8)
    np.testing.assert_allclose(const[:3, 3:], 0.0, atol=1e-8)
    np.testing.assert_allclose(const[3:, :3], 0.0, atol=1e-8)
    np.testing.assert_allclose(const[3:, 3:], np.eye(3) * 149.55595257029483, atol=1e-8)
