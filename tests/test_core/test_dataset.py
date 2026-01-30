"""Tests of dataset classes and functions."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.dataset import (
    Dataset,
    DatasetList,
    set_datasets_from_multiple_filesets,
    set_datasets_from_single_fileset,
    set_datasets_from_structures,
)

cwd = Path(__file__).parent
path_files = str(cwd) + "/../files/"


def test_dataset_vasp1(params_MgO):
    """Test for Dataset class."""
    files = [
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
    ]
    dataset = Dataset(dataset_type="vasp", files=files, name="data1")
    dataset.parse_files(params_MgO)

    assert dataset.include_force
    assert dataset.include_stress
    assert dataset.weight == pytest.approx(1.0)
    assert dataset.name == "data1"
    assert dataset.files == files
    np.testing.assert_allclose(dataset.energies, [-321.77619112, -321.77563085])
    assert len(dataset.forces) == 384
    assert len(dataset.stresses) == 12
    assert len(dataset.volumes) == 2
    assert len(dataset.structures) == 2
    assert len(dataset.total_n_atoms) == 2
    assert dataset.exist_force
    assert dataset.exist_stress


def test_dataset_vasp2(params_MgO):
    """Test for Dataset class."""
    location = path_files + "vasprun-*-MgO.xml"
    dataset = Dataset(dataset_type="vasp", location=location, name="data1")
    assert len(dataset.files) == 2

    train, test = dataset.split_files(train_ratio=0.5)
    assert len(train.files) == 1
    assert len(test.files) == 1

    dataset.parse_files(params_MgO)
    assert len(dataset.energies) == 2


def test_dataset_vasp3(params_MgO):
    """Test for Dataset class."""
    strings = [path_files + "vasprun-*-MgO.xml", "False", "0.1"]
    dataset = Dataset(dataset_type="vasp", strings=strings, name="data1")
    assert not dataset.include_force
    assert not dataset.include_stress
    assert dataset.weight == pytest.approx(0.1)
    assert len(dataset.files) == 2

    dataset.parse_files(params_MgO)
    assert len(dataset.energies) == 2


def test_dataset_phono3py(regdata_mp_149):
    """Test for Dataset class."""
    params, _ = regdata_mp_149
    files = path_files + "phonopy_training_dataset.yaml.xz"
    dataset = Dataset(dataset_type="phono3py", files=files, name="data1")
    dataset.parse_files(params)

    assert len(dataset.energies) == 200
    assert len(dataset.forces) == 38400
    assert len(dataset.stresses) == 1200
    assert len(dataset.volumes) == 200
    assert len(dataset.structures) == 200
    assert len(dataset.total_n_atoms) == 200

    train, test = dataset.split_dft()
    assert len(train.energies) == 180
    assert len(train.forces) == 34560
    assert len(train.stresses) == 1080
    assert len(train.volumes) == 180
    assert len(train.structures) == 180
    assert len(train.total_n_atoms) == 180

    assert len(test.energies) == 20
    assert len(test.forces) == 3840
    assert len(test.stresses) == 120
    assert len(test.volumes) == 20
    assert len(test.structures) == 20
    assert len(test.total_n_atoms) == 20

    dataset.sort_dft()
    data_sliced = dataset.slice_dft(5, 8)
    assert len(data_sliced.energies) == 3
    assert len(data_sliced.forces) == 576
    assert len(data_sliced.stresses) == 18
    assert len(data_sliced.volumes) == 3
    assert len(data_sliced.structures) == 3
    assert len(data_sliced.total_n_atoms) == 3


def test_dataset_sscha():
    """Test for Dataset class."""
    files = [
        path_files + "sscha_results_0001.yaml",
        path_files + "sscha_results_0002.yaml",
        path_files + "sscha_results_0003.yaml",
    ]
    dataset = Dataset(dataset_type="sscha", files=files, name="data1")

    assert dataset.include_force
    assert dataset.include_stress
    assert dataset.weight == pytest.approx(1.0)
    assert dataset.files == files


def test_dataset_electron():
    """Test for Dataset class."""
    location = path_files + "electron-*.yaml"
    dataset = Dataset(dataset_type="electron", location=location, name="data1")

    assert dataset.include_force
    assert dataset.include_stress
    assert dataset.weight == pytest.approx(1.0)
    assert len(dataset.files) == 3


def test_dataset_list(params_MgO):
    """Test DatasetList."""
    location = path_files + "vasprun-*-MgO.xml"
    dataset1 = Dataset(dataset_type="vasp", location=location, name="data1")

    datasets = DatasetList()
    datasets.append([dataset1, dataset1])
    datasets.append(dataset1)
    assert len(datasets) == 3

    datasets = DatasetList(dataset1)
    datasets.append([dataset1, dataset1])
    assert len(datasets) == 3

    datasets = DatasetList([dataset1, dataset1])
    datasets.append(dataset1)
    assert len(datasets) == 3

    datasets.parse_files(params_MgO)
    for ds in datasets:
        assert len(ds.energies) == 2

    assert isinstance(datasets.datasets, list)
    assert datasets.include_force
    assert len(datasets[0].energies) == 2


def test_set_datasets_from_multiple_filesets(params_MgO):
    """Test set_datasets_from_multiple_filesets."""
    files = [
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
    ]
    train = [files, files]
    test = [files, files]
    train_all, test_all = set_datasets_from_multiple_filesets(params_MgO, train, test)

    for ds in train_all:
        assert len(ds.energies) == 2
    for ds in test_all:
        assert len(ds.energies) == 2


def test_set_datasets_from_single_fileset1(params_MgO):
    """Test set_datasets_from_single_fileset."""
    files = [
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
    ]
    train_all, test_all = set_datasets_from_single_fileset(
        params_MgO, files=files, train_ratio=0.9
    )
    for ds in train_all:
        assert len(ds.energies) == 9
    for ds in test_all:
        assert len(ds.energies) == 1


def test_set_datasets_from_single_fileset2(params_MgO):
    """Test set_datasets_from_single_fileset."""
    train_files = [
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
    ]
    test_files = [
        path_files + "vasprun-00001-MgO.xml",
        path_files + "vasprun-00002-MgO.xml",
    ]
    train_all, test_all = set_datasets_from_single_fileset(
        params_MgO,
        train_files=train_files,
        test_files=test_files,
    )
    for ds in train_all:
        assert len(ds.energies) == 8
    for ds in test_all:
        assert len(ds.energies) == 2


def test_set_datasets_from_single_fileset3(regdata_mp_149):
    """Test set_datasets_from_single_fileset."""
    params, _ = regdata_mp_149
    files = path_files + "phonopy_training_dataset.yaml.xz"
    train_all, test_all = set_datasets_from_single_fileset(params, files=files)
    for ds in train_all:
        assert len(ds.energies) == 180
    for ds in test_all:
        assert len(ds.energies) == 20


def test_set_datasets_from_structures1(params_MgO, structure_rocksalt):
    """Test set_datasets_from_structures."""
    structures = [structure_rocksalt] * 10
    energies = np.random.random(10)
    forces = np.random.random((10, 3, 8))
    stresses = np.random.random((10, 3, 3))
    train_all, test_all = set_datasets_from_structures(
        params_MgO,
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=stresses,
    )
    for ds in train_all:
        assert len(ds.energies) == 9
    for ds in test_all:
        assert len(ds.energies) == 1


def test_set_datasets_from_structures2(params_MgO, structure_rocksalt):
    """Test set_datasets_from_structures."""
    structures = [structure_rocksalt] * 10
    energies = np.random.random(10)
    forces = np.random.random((10, 3, 8))
    stresses = np.random.random((10, 3, 3))
    train_all, test_all = set_datasets_from_structures(
        params_MgO,
        train_structures=structures,
        train_energies=energies,
        train_forces=forces,
        train_stresses=stresses,
        test_structures=structures,
        test_energies=energies,
        test_forces=forces,
        test_stresses=stresses,
    )
    for ds in train_all:
        assert len(ds.energies) == 10
    for ds in test_all:
        assert len(ds.energies) == 10

    shift = sum(params_MgO.atomic_energy) * 4
    for ds in train_all:
        np.testing.assert_allclose(ds.energies, energies - shift)

    # In this case, atomic energies are not subtracted in the following command
    # because atomic energies were subtracted in set_datasets_from_structures.
    train_all.subtract_atomic_energy(params_MgO.atomic_energy)
    for ds in train_all:
        np.testing.assert_allclose(ds.energies, energies - shift)
