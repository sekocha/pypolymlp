"""Tests disorder.py."""

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
