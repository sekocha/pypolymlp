"""Tests of calculating distributions from SSCHA results."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.sscha.sscha_distribution import SSCHADistribution

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_sscha_distribution():
    """Test SSCHADistribution."""
    path_sscha = path_file + "others/sscha_restart/"
    yaml = path_sscha + "sscha_results.yaml"
    fc2hdf5 = path_sscha + "fc2.hdf5"

    distrib = SSCHADistribution(yamlfile=yaml, fc2file=fc2hdf5, pot=pot, verbose=True)
    distrib.set_structure_distribution(n_samples=100)
    assert distrib.displacements.shape == (100, 3, 32)
    assert distrib.forces.shape == (100, 3, 32)
    assert distrib.energies.shape == (100,)
    assert distrib.static_potential == pytest.approx(-1322.92893961425)
    assert np.sum(distrib.static_forces) == pytest.approx(0.0)
    assert distrib.average_forces.shape == (3, 32)
    assert len(distrib.supercells) == 100

    distrib.save_structure_distribution(path="tmp")
    shutil.rmtree("tmp")
