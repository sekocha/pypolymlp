"""Tests of polymlp parser."""

import os
from pathlib import Path

from pypolymlp.core.io_polymlp import (
    convert_to_yaml,
    find_mlps,
    is_hybrid,
    is_legacy,
    load_mlp,
    load_mlps,
)

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
