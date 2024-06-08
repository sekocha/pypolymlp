#!/usr/bin/env python
import argparse
import os

from lammps_api.sscha.restart import load_restart_from_result_yaml
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN

# from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sscha_file",
        type=str,
        default="sscha_results.yaml",
        help="Location of sscha_results.yaml file",
    )
    parser.add_argument("--born", type=str, default=None, help="Location of BORN")
    args = parser.parse_args()

    sscha = load_restart_from_result_yaml(args.sscha_file)
    phonon = sscha.get_phonopy_obj()
    fc = phonon.force_constants

    filedir = "/".join(os.path.abspath(args.sscha_file).split("/")[:-1])
    print(filedir)
    mesh = [20, 20, 20]
    datname = filedir + "/total_dos_nac.dat"

    unitcell = phonon.unitcell
    supercell = phonon.supercell
    supercell_matrix = phonon.get_supercell_matrix()
    # primitive = np.eye(3)
    # primitive = [[-0.5,0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,-0.5]]

    phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
    # primitive_matrix=primitive)
    if args.born is not None:
        nac_params = parse_BORN(phonon.primitive, filename=args.born)
        nac_params["factor"] = 14.400
        phonon = Phonopy(
            unitcell,
            supercell_matrix=supercell_matrix,
            #    primitive_matrix=primitive,
            nac_params=nac_params,
        )

    phonon.force_constants = fc
    phonon.run_mesh(mesh, with_eigenvectors=False, is_mesh_symmetry=True)
    mesh_dict = phonon.get_mesh_dict()
    phonon.run_total_dos()
    phonon.write_total_dos(filename=datname)
