"""Tests of polynomial MLP development using API"""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.dataset import DatasetList
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent
path_files = str(cwd) + "/../../files/"


def test_attrs():
    """Test attributes and functions."""
    infile = str(cwd / "polymlp.in.phono3py.Si")
    polymlp = Pypolymlp()
    polymlp.load_parameter_file(infile, prefix=str(cwd))
    polymlp.fit()
    polymlp.estimate_error(log_energy=False)

    polymlp.print_params()

    output_tmp = str(cwd) + "/tmp"
    polymlp.save_mlp(filename=output_tmp)
    polymlp.save_params(filename=output_tmp)
    polymlp.save_errors(filename=output_tmp)

    os.remove(output_tmp)

    mlp_model = polymlp.summary
    np.testing.assert_allclose(mlp_model.scaled_coeffs, polymlp.coeffs)
    assert polymlp.parameters.n_type == 1
    assert polymlp.n_features == 168
    assert isinstance(polymlp.train, DatasetList)
    assert isinstance(polymlp.test, DatasetList)
    assert isinstance(polymlp.datasets[0], DatasetList)
    assert isinstance(polymlp.datasets[1], DatasetList)


def test_load_mlp():
    """Test for loading polymlp files."""
    # Parse polymlp.lammps.pair
    filename = cwd / "mlps/polymlp.lammps.pair"
    coeff_true = 9.352307613515078e00 / 2.067583465937491e-01

    mlp = Pypolymlp()
    with open(filename, "rt") as fp:
        mlp.load_mlp(fp)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    mlp = Pypolymlp()
    mlp.load_mlp(filename)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    # Parse polymlp.yaml.gtinv
    filename = cwd / "mlps/polymlp.yaml.gtinv"
    coeff_true = 5.794375827500248e01

    mlp = Pypolymlp()
    with open(filename, "rt") as fp:
        mlp.load_mlp(fp)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)

    mlp = Pypolymlp()
    mlp.load_mlp(filename)
    assert mlp.coeffs[0] == pytest.approx(coeff_true, rel=1e-8)


def test_split_train_test():
    """Test for splitting dataset."""
    list1 = np.arange(50)
    mlp = Pypolymlp()
    train, test = mlp.split_train_test(list1)
    assert len(train) == 45
    assert len(test) == 5
    train, test = mlp.split_train_test(list1, train_ratio=0.8)
    assert len(train) == 40
    assert len(test) == 10


def test_convert_to_yaml():
    """Test for converting legacy polymlp file."""
    mlp = Pypolymlp()
    mlp.convert_to_yaml(
        cwd / "mlps/polymlp.lammps.pair",
        cwd / "mlps/polymlp_tmp.yaml",
    )
    os.remove(cwd / "mlps/polymlp_tmp.yaml")


def test_get_structures_from_poscars():
    """Test get_structures_from_poscars."""
    mlp = Pypolymlp()
    poscar = path_files + "POSCAR-rocksalt"
    st = mlp.get_structures_from_poscars(poscar)
    np.testing.assert_equal(st.n_atoms, [4, 4])

    poscars = [poscar, poscar]
    st = mlp.get_structures_from_poscars(poscars)
    np.testing.assert_equal(st[0].n_atoms, [4, 4])
    np.testing.assert_equal(st[1].n_atoms, [4, 4])


# TODO
# def test_load_mlps_hybrid():
#     """Test for loading hybrid polymlp files."""
#     pass
