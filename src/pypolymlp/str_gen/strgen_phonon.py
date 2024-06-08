#!/usr/bin/env python
import argparse
import os

from pypolymlp.core.displacements import generate_random_const_displacements
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.vasp_utils import write_poscar_file
from pypolymlp.utils.yaml_utils import save_cells


def run_strgen_phonon(
    filename,
    supercell_size=[2, 2, 2],
    n_samples=20,
    displacements=0.03,
    use_phonopy=True,
):

    unitcell = Poscar(filename).get_structure()
    if use_phonopy:
        from pypolymlp.utils.phonopy_utils import phonopy_supercell

        supercell = phonopy_supercell(
            unitcell, supercell_diag=supercell_size, return_phonopy=False
        )
    else:
        supercell = supercell_diagonal(unitcell, size=supercell_size)

    _, st_dicts = generate_random_const_displacements(
        supercell, n_samples=n_samples, displacements=displacements
    )

    os.makedirs("poscars_phonon", exist_ok=True)
    write_poscar_file(supercell, filename="poscars_phonon/poscar-00000")
    for i, st in enumerate(st_dicts):
        write_poscar_file(
            st,
            filename="poscars_phonon/poscar-" + str(i + 1).zfill(5),
            header="pypolymlp: disp.-" + str(i + 1).zfill(5),
        )

    save_cells(unitcell, supercell, filename="polymlp_str.yaml")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        required=True,
        help="Initial structures in POSCAR format",
    )
    args = parser.parse_args()

    run_strgen_phonon(
        args.poscar,
        supercell_size=[3, 3, 2],
        n_samples=20,
        displacements=0.03,
    )
