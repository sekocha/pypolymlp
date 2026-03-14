"""Tests of io_polymlp."""

import os
from pathlib import Path

import numpy as np

from pypolymlp.core.io_polymlp import (
    convert_to_yaml,
    find_mlps,
    is_hybrid,
    is_legacy,
    load_mlp,
    load_mlps,
    save_mlp,
    save_mlps,
)
from pypolymlp.core.params import PolymlpParams

cwd = Path(__file__).parent


def test_save_mlp():
    """Test for save_mlp and save_mlps."""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    params_single, coeffs = load_mlp(file1)
    scales = np.ones(len(coeffs))
    save_mlp(params_single, coeffs, scales, filename="tmp.yaml")
    os.remove("tmp.yaml")

    file2 = str(cwd) + "/../files/polymlp.yaml.MgO"
    params_single, coeffs = load_mlp(file2)
    scales = np.ones(len(coeffs))
    save_mlp(params_single, coeffs, scales, filename="tmp.yaml")
    os.remove("tmp.yaml")

    params = PolymlpParams([params_single, params_single])
    cumulative_n_features = [len(coeffs), len(coeffs) * 2]
    coeffs_all = np.tile(coeffs, 2)
    scales_all = np.tile(scales, 2)
    save_mlps(
        params,
        coeffs_all,
        scales_all,
        cumulative_n_features,
        filename="tmp.yaml",
    )
    os.remove("tmp.yaml.1")
    os.remove("tmp.yaml.2")


def test_load_mlp():
    """Test for load_mlp."""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    params_single, coeffs = load_mlp(file1)
    assert len(coeffs) == 12694

    file2 = str(cwd) + "/../files/polymlp.yaml.MgO"
    params_single, coeffs = load_mlp(file2)
    assert len(coeffs) == 1899
    assert params_single.n_type == 2

    params, coeffs_array = load_mlps([file1, file1])
    assert len(coeffs_array[0]) == 12694
    assert len(coeffs_array[1]) == 12694
    assert len(params) == 2

    params, coeffs_array = load_mlps([file1])
    assert len(coeffs_array[0]) == 12694
    assert len(params) == 1


def test_find_mlps():
    """Test for find_mlps"""
    mlps = find_mlps(str(cwd) + "/../files")
    assert len(mlps) == 2


def test_convert_to_yaml():
    """Test convert_to_yaml"""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    convert_to_yaml(file1, yaml="tmp.yaml")
    os.remove("tmp.yaml")

    files = [file1, file1]
    convert_to_yaml(files, yaml="tmp.yaml")
    os.remove("tmp.yaml.1")
    os.remove("tmp.yaml.2")

    file2 = str(cwd) + "/../files/polymlp.yaml.MgO"
    convert_to_yaml(file2, yaml="tmp.yaml")
    files = [file2, file2]
    convert_to_yaml(files, yaml="tmp.yaml")


def test_is_legacy():
    """Test for is_legacy"""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    assert is_legacy(file1) == True
    file2 = str(cwd) + "/../files/polymlp.yaml.MgO"
    assert is_legacy(file2) == False


def test_is_hybrid():
    """Test for is_hybrid"""
    file1 = str(cwd) + "/../files/polymlp.lammps.Ti"
    assert is_hybrid(file1) == False
    files = [file1]
    assert is_hybrid(files) == False
    files = [file1, file1]
    assert is_hybrid(files) == True
