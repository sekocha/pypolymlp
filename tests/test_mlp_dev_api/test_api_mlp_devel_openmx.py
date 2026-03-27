"""Tests of polynomial MLP development using API and openmx data."""

import os
from pathlib import Path

from test_mlp_devel_openmx import _assert_AgC, _assert_AgC_hybrid

from pypolymlp.core.interface_openmx import parse_openmx
from pypolymlp.core.utils import split_ids_train_test
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent

# Parse openmx data files.
datafiles = [str(cwd) + "/data-openmx-AgC/sample.md"]
structures, energies, forces = parse_openmx(datafiles)

# Split dataset into training and test datasets automatically.
n_data = len(energies)
train_ids, test_ids = split_ids_train_test(n_data, train_ratio=0.9)
train_structures = [structures[i] for i in train_ids]
test_structures = [structures[i] for i in test_ids]
train_energies = energies[train_ids]
test_energies = energies[test_ids]
train_forces = [forces[i] for i in train_ids]
test_forces = [forces[i] for i in test_ids]


def test_mlp_dev_openmx():
    """Test MLP development using openmx data."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Ag", "C"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=(4, 4),
        reg_alpha_params=(-5, -1, 5),
        gaussian_params2=(0.0, 5.0, 7),
        atomic_energy_unit="Hartree",
        atomic_energy=(-114.017822031176, -5.499002028934),
    )
    polymlp.set_datasets_structures(
        train_structures=train_structures,
        test_structures=test_structures,
        train_energies=train_energies,
        test_energies=test_energies,
        train_forces=train_forces,
        test_forces=test_forces,
    )
    polymlp.print_params()
    polymlp.run(verbose=True)
    polymlp.save_mlp(filename="polymlp.yaml")
    os.remove("polymlp.yaml")

    assert polymlp.n_features == 1660
    tag_train = "data1"
    tag_test = "data2"
    error_train = polymlp.summary.error_train[tag_train]
    error_test = polymlp.summary.error_test[tag_test]
    _assert_AgC(error_train, error_test)


def test_mlp_dev_openmx_hybrid():
    """Test MLP development (hybrid) using openmx data."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Ag", "C"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=(4, 4),
        reg_alpha_params=(-5, -1, 5),
        gaussian_params2=(0.0, 5.0, 7),
        atomic_energy_unit="Hartree",
        atomic_energy=(-114.017822031176, -5.499002028934),
    )
    polymlp.append_hybrid_params(
        elements=["Ag", "C"],
        cutoff=4.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=(6, 6),
        gaussian_params2=(0.0, 3.0, 5),
    )
    polymlp.set_datasets_structures(
        train_structures=train_structures,
        test_structures=test_structures,
        train_energies=train_energies,
        test_energies=test_energies,
        train_forces=train_forces,
        test_forces=test_forces,
    )
    polymlp.print_params()
    polymlp.run(verbose=True)
    polymlp.save_mlp(filename="polymlp.yaml")
    os.remove("polymlp.yaml.1")
    os.remove("polymlp.yaml.2")

    assert polymlp.n_features == 4099
    tag_train = "data1"
    tag_test = "data2"
    error_train = polymlp.summary.error_train[tag_train]
    error_test = polymlp.summary.error_test[tag_test]
    _assert_AgC_hybrid(error_train, error_test)
