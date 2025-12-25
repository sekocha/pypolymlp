"""Tests of openmx parser."""

from pathlib import Path

import numpy as np

from pypolymlp.core.interface_yaml import (  # extract_electron_properties,; parse_electron_yamls,
    parse_sscha_yamls,
)

cwd = Path(__file__).parent


def test_parse_sscha_yamls():
    """Test for parse_sscha_yamls."""
    yamls = [
        cwd / "./../files/sscha_results_0001.yaml",
        cwd / "./../files/sscha_results_0002.yaml",
        cwd / "./../files/sscha_results_0003.yaml",
    ]
    strs, free_energies, forces = parse_sscha_yamls(yamls)
    np.testing.assert_allclose(free_energies, [-1.41748102, -0.94484543], atol=1e-6)
    assert len(strs) == 2
    assert len(forces) == 2
    assert forces[0].shape == (3, 40)
    assert forces[1].shape == (3, 40)
