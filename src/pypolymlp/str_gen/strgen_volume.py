#!/usr/bin/env python
import argparse
import os

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import multiple_isotropic_volume_changes
from pypolymlp.utils.vasp_utils import write_poscar_file


def run_strgen_volume(filename, eps_min=0.8, eps_max=2.0, n_eps=10):

    st_dict = Poscar(filename).get_structure()
    st_dicts = multiple_isotropic_volume_changes(
        st_dict, eps_min=eps_min, eps_max=eps_max, n_eps=n_eps
    )

    os.makedirs("poscars_volume", exist_ok=True)
    for i, st in enumerate(st_dicts):
        write_poscar_file(
            st,
            filename="poscars_volume/poscar-" + str(i + 1).zfill(3),
            header="pypolymlp: volume-" + str(i + 1).zfill(3),
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default="POSCAR",
        help="poscar file name",
    )
    parser.add_argument("--min", type=float, default=0.8, help="Minimum volume ratio")
    parser.add_argument("--max", type=float, default=1.2, help="Maximum volume ratio")
    parser.add_argument(
        "-n", "--n_volumes", type=int, default=15, help="Number of volumes"
    )
    args = parser.parse_args()

    run_strgen_volume(
        args.poscar,
        eps_min=args.min,
        eps_max=args.max,
        n_eps=args.n_volumes,
    )
