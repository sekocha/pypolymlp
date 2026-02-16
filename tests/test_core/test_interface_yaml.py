"""Tests of yaml parser."""

from pathlib import Path

import numpy as np

from pypolymlp.core.interface_yaml import (
    extract_electron_properties,
    parse_electron_yamls,
    parse_sscha_yamls,
    split_imaginary,
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
    np.testing.assert_allclose(
        free_energies,
        [-1.41748102, -1.803403, -0.94484543],
        atol=1e-6,
    )
    assert len(strs) == 3
    assert len(forces) == 3
    assert forces[0].shape == (3, 40)
    assert forces[1].shape == (3, 40)
    assert forces[2].shape == (3, 40)


def test_split_imaginary():
    """Test for split_imaginary."""
    yamls = [
        cwd / "./../files/sscha_results_0001.yaml",
        cwd / "./../files/sscha_results_0002.yaml",
        cwd / "./../files/sscha_results_0003.yaml",
    ]
    yaml1, yaml2 = split_imaginary(yamls)
    assert len(yaml1) == 2
    assert len(yaml2) == 1


def test_electron_yamls():
    """Test for parsing electron.yaml."""
    yamls = [
        cwd / "./../files/electron-0.yaml",
        cwd / "./../files/electron-1.yaml",
        cwd / "./../files/electron-2.yaml",
    ]
    data = parse_electron_yamls(yamls)
    strs, props = extract_electron_properties(
        data, temperature=500.0, target="free_energy"
    )
    np.testing.assert_allclose(
        props, [-0.00537118, -0.00520776, -0.00521567], atol=1e-6
    )
