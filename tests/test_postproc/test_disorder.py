"""Tests disorder.py."""

import shutil
from pathlib import Path

from pypolymlp.postproc.disorder import PolymlpDisorder

cwd = Path(__file__).parent
path_files = str(cwd) + "/../files/"


def test_PolymlpDisorder():
    """Test PolymlpDisorder."""
    pot = path_files + "polymlp.yaml.MgO"
    lattice_poscar = path_files + "POSCAR-SC"

    occ = [[("Mg", 0.5), ("O", 0.5)]]
    disorder = PolymlpDisorder(
        occupancy=occ,
        pot=pot,
        lattice_poscar=lattice_poscar,
        supercell_size=(2, 2, 2),
    )

    disorder.set_displaced_lattices(
        n_samples=2,
        max_distance=1.0,
        include_base_structure=False,
    )
    assert len(disorder.structures) == 2

    disorder.set_displaced_lattices(
        n_samples=2,
        max_distance=1.0,
        include_base_structure=True,
    )
    assert len(disorder.structures) == 3

    disorder.eval_random_properties(n_samples=2, max_iter=1)

    e, f, s = disorder.properties
    assert e.shape == (3,)
    assert f.shape == (3, 3, 64)
    assert s.shape == (3, 6)

    disorder.save_properties(path="tmp")
    shutil.rmtree("tmp")
