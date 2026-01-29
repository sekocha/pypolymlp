"""Tests of phono3py parser."""

from pathlib import Path

import numpy as np
import pytest
from phono3py.api_phono3py import Phono3py

from pypolymlp.core.interface_phono3py import (
    Phono3pyYaml,
    parse_phono3py_yaml,
    parse_phono3py_yaml_fcs,
    parse_structures_from_phono3py_yaml,
)

cwd = Path(__file__).parent


def test_phono3py_yaml_class(phono3py_mp_149):
    """Test Phono3pyYaml class."""
    ph3yaml = Phono3pyYaml(phono3py_mp_149, use_phonon_dataset=False)
    assert ph3yaml.displacements.shape == (200, 3, 64)
    assert ph3yaml.forces.shape == (200, 3, 64)
    assert len(ph3yaml.supercells) == 200
    assert len(ph3yaml.energies) == 200

    np.testing.assert_equal(ph3yaml.phonon_dataset[0], ph3yaml.displacements)
    np.testing.assert_equal(ph3yaml.phonon_dataset[1], ph3yaml.forces)

    assert ph3yaml.supercell_phono3py.cell[0, 0] == pytest.approx(10.86912324)
    assert isinstance(ph3yaml.phono3py, Phono3py) == True


def test_parse_phono3py_yaml(phono3py_mp_149):
    """Test parse_phono3py_yaml."""
    dft = parse_phono3py_yaml(phono3py_mp_149, element_order=["Si"])
    assert len(dft.energies) == 200
    assert dft.forces.shape[0] == (38400)

    dft, disps = parse_phono3py_yaml(
        phono3py_mp_149, element_order=["Si"], return_displacements=True
    )
    assert disps.shape == (200, 3, 64)


def test_parse_phono3py_yaml_fcs(phono3py_mp_149):
    """Test parse_phono3py_yaml_fcs."""
    cell, disps, supercells = parse_phono3py_yaml_fcs(phono3py_mp_149)
    assert disps.shape == (200, 3, 64)
    assert len(supercells) == 200


def test_parse_structures_from_phono3py_yaml(phono3py_mp_149):
    """Test parse_structures_from_phono3py_yaml."""
    supercells = parse_structures_from_phono3py_yaml(phono3py_mp_149)
    assert len(supercells) == 200
