"""Tests of calculating distributions from SSCHA results."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.sscha.sscha_distribution import SSCHADistribution

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_sscha_distribution(unitcell_mlp_Al):
    """Test SSCHADistribution."""
    _, pot, _ = unitcell_mlp_Al
    path_sscha = path_file + "others/sscha_restart/"
    yaml = path_sscha + "sscha_results.yaml"
    fc2hdf5 = path_sscha + "fc2.hdf5"

    distrib = SSCHADistribution(yamlfile=yaml, fc2file=fc2hdf5, pot=pot, verbose=True)
    distrib.run_structure_distribution(n_samples=100)
    assert len(distrib.unitcell.elements) == 4
    assert len(distrib.supercell.elements) == 32
    assert distrib.displacements.shape == (100, 3, 32)
    assert distrib.forces.shape == (100, 3, 32)
    assert distrib.energies.shape == (100,)
    assert distrib.static_potential == pytest.approx(-1322.92893961425)
    assert np.sum(distrib.static_forces) == pytest.approx(0.0)
    assert distrib.average_forces.shape == (3, 32)
    assert len(distrib.supercells) == 100

    distrib.save_structure_distribution(path="tmp")
    shutil.rmtree("tmp")

    assert distrib.polymlp.split("/")[-1] == "polymlp.yaml.gtinv.Al"
    assert distrib.temperature == 700
    assert distrib.parameters["temperature"] == 700
    assert len(distrib.logs) == 4
    assert distrib.delta_fc < 0.01
    assert distrib.converge
    assert not distrib.imaginary
    assert distrib.force_constants.shape == (32, 32, 3, 3)
    assert len(distrib.unitcell.elements) == 4
    assert len(distrib.supercell.elements) == 32
    np.testing.assert_equal(distrib.supercell_matrix, np.diag((2, 2, 2)))
    assert distrib.volume == pytest.approx(65.77091478008525)
