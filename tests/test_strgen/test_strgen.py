"""Tests of functions for generating DFT structures."""

import os
import shutil
from pathlib import Path

import numpy as np

from pypolymlp.api.pypolymlp_str import PypolymlpStructureGenerator
from pypolymlp.core.strgen import (
    set_structure_id,
    set_volume_eps_array,
    write_structures,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/"
file_rs = path_file + "POSCAR-rocksalt"
file_wz = path_file + "POSCAR-WZ"


def test_functions():
    """Test write_structures and set_structure_id."""
    polymlp = PypolymlpStructureGenerator(verbose=True)
    polymlp.load_poscars([file_rs, file_wz])
    polymlp.build_supercells_auto()
    polymlp.run_standard_algorithm(n_samples=2, max_distance=1.0)

    structures = set_structure_id(polymlp.sample_structures, "Test-POSCARs", "Test")
    for st in structures:
        assert st.base == "Test-POSCARs"
        assert st.mode == "Test"

    base_info = []
    for gen in polymlp._strgen_instances:
        base_dict = {
            "id": gen.name,
            "size": list(gen.supercell_size),
            "n_atoms": list(gen.n_atoms),
        }
        base_info.append(base_dict)

    write_structures(
        polymlp.sample_structures,
        base_info,
        path="tmp",
    )
    os.remove("polymlp_str_samples.yaml")
    shutil.rmtree("tmp")


def test_set_volume_eps_array():
    """Test set_volume_eps_array."""
    eps_array = set_volume_eps_array(
        n_samples=8,
        eps_min=0.7,
        eps_max=1.4,
        dense_equilibrium=False,
    )
    np.testing.assert_allclose(eps_array, [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    eps_array = set_volume_eps_array(
        n_samples=8,
        eps_min=0.7,
        eps_max=1.4,
        dense_equilibrium=True,
    )
    true = [
        0.7,
        0.9,
        0.92222222,
        0.94444444,
        0.96666667,
        0.98888889,
        1.01111111,
        1.03333333,
        1.05555556,
        1.07777778,
        1.1,
        1.4,
    ]
    np.testing.assert_allclose(eps_array, true)


# def test_run_density_algorithm():
#     """Test run_density_algorithm."""
#     polymlp = PypolymlpStructureGenerator(verbose=True)
#     polymlp.load_poscars([file_rs, file_wz])
#     polymlp.build_supercells_auto()
#     polymlp.run_density_algorithm(n_samples=2, vol_algorithm="low_auto")
#     polymlp.run_density_algorithm(n_samples=2, vol_algorithm="high_auto")
#     assert polymlp.n_samples == 8
#
#     assert len(polymlp.sample_structures[0].elements) == 64
#     assert len(polymlp.sample_structures[-1].elements) == 72
