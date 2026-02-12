"""Tests of API for generating DFT structures."""

import os
import shutil
from pathlib import Path

from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/"
file_rs = path_file + "POSCAR-rocksalt"


def test_run_const_displacements():
    """Test for run_const_displacements."""
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_poscars(file_rs)
    polymlp.build_supercell(supercell_size=(2, 2, 2))
    polymlp.run_const_displacements(n_samples=2, distance=0.1)
    polymlp.save_random_structures(path="tmp")
    os.remove("polymlp_str_samples.yaml")
    shutil.rmtree("tmp")

    polymlp.load_poscars([file_rs, file_rs])
    polymlp.build_supercell(supercell_size=(2, 2, 2))
    polymlp.run_const_displacements(n_samples=2, distance=0.1)

    assert len(polymlp.sample_structures) == 6
    assert polymlp.n_samples == 6


def test_run_sequential_displacements():
    """Test for run_sequential_displacements."""
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_structures_from_files(poscars=file_rs)
    polymlp.build_supercell(supercell_size=(2, 2, 2))
    polymlp.run_sequential_displacements(
        n_samples=2,
        distance_lb=0.01,
        distance_ub=0.5,
        n_volumes=1,
    )
    assert polymlp.n_samples == 2

    polymlp.load_structures_from_files(poscars=[file_rs, file_rs])
    polymlp.build_supercell(supercell_size=(2, 2, 2))
    polymlp.run_sequential_displacements(
        n_samples=2,
        distance_lb=0.1,
        distance_ub=0.5,
        n_volumes=3,
    )
    assert polymlp.n_samples == 14


def test_run_isotropic_volume_changes():
    """Test for run_isotropic_volume_changes."""
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_poscars([file_rs, file_rs])
    polymlp.build_supercell(supercell_size=(2, 2, 2))
    polymlp.run_isotropic_volume_changes(n_samples=3, dense_equilibrium=False)
    assert polymlp.n_samples == 6

    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_poscars([file_rs, file_rs])
    polymlp.build_supercell(supercell_size=(2, 2, 2))
    polymlp.run_isotropic_volume_changes(n_samples=3, dense_equilibrium=True)
    assert polymlp.n_samples == 10


def test_run_standard_algorithm():
    """Test run_standard_algorithm."""
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_poscars([file_rs, file_rs])
    polymlp.build_supercells_auto()
    polymlp.run_standard_algorithm(n_samples=2, max_distance=1.0)
    assert polymlp.n_samples == 4
    for st in polymlp.sample_structures:
        assert len(st.elements) == 64


def test_run_density_algorithm():
    """Test run_density_algorithm."""
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_poscars([file_rs, file_rs])
    polymlp.build_supercells_auto()
    polymlp.run_density_algorithm(n_samples=2, vol_algorithm="low_auto")
    polymlp.run_density_algorithm(n_samples=2, vol_algorithm="high_auto")
    assert polymlp.n_samples == 8
