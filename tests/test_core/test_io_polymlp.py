"""Tests of polymlp parser."""

from pathlib import Path

from pypolymlp.core.io_polymlp import find_mlps, load_mlp, load_mlps

cwd = Path(__file__).parent


def test_load_mlp():
    """Test for load_mlp."""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    params, coeffs = load_mlp(file1)
    assert len(coeffs) == 12694

    file2 = str(cwd) + "/../files/polymlp.yaml.MgO"
    params, coeffs = load_mlp(file2)
    assert len(coeffs) == 1899

    params_array, coeffs_array = load_mlps([file1, file2])
    len(coeffs_array[0]) == 12694
    len(coeffs_array[1]) == 1899


def test_find_mlps():
    """Test for find_mlps"""
    mlps = find_mlps(str(cwd) + "/../files")
    assert len(mlps) == 1
