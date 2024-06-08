#!/usr/bin/env python
import argparse
import os

import numpy as np
from lammps_api.sscha.restart import load_restart_from_result_yaml
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sscha_file",
        type=str,
        default="sscha_results.yaml",
        help="Location of sscha_results.yaml file",
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["bcc", "fcc", "rocksalt", "perovskite", "perovskite2"],
        required=True,
        help="Structure type",
    )
    parser.add_argument("--harmonic", action="store_true", help="Harmonic calculation")
    parser.add_argument(
        "--pot",
        type=str,
        default="mlp.lammps",
        help="Location of mlp.lammps",
    )
    args = parser.parse_args()

    path = None
    if args.structure == "bcc":
        primitive = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
        path = [
            [
                [0, 0, 0],
                [0.5, -0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0, 0, 0],
                [0, 0.0, 0.5],
            ]
        ]
        labels = ["G", "H", "P", "G", "N"]
    elif args.structure == "fcc":
        primitive = [[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]]
        path = [
            [[0, 0, 0], [0.5, 0.5, 0]],
            [
                [0.5, 0.5, 1],
                [0.375, 0.375, 0.75],
                [0, 0, 0],
                [0.5, 0.5, 0.5],
            ],
        ]
        labels = ["G", "X", "K", "G", "L"]
    elif args.structure == "rocksalt":
        primitive = [[0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]]
        path = [[[0.5, 0.5, 0], [0, 0, 0]], [[0, 0, 0], [0.5, 0.5, 0.5]]]
        labels = ["X", "G", "L"]
    elif args.structure == "perovskite":
        primitive = np.eye(3)
        path = [
            [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]],
        ]
        labels = ["G", "X", "M", "G", "M", "R"]
    elif args.structure == "perovskite2":
        primitive = np.eye(3)
        path = [
            [[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0]],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
        ]
        labels = ["G", "X", "M", "G", "R", "M"]

    if path is not None:
        sscha = load_restart_from_result_yaml(args.sscha_file)
        phonon = sscha.get_phonopy_obj()

        if args.harmonic:
            sscha.set_pot(args.pot)
            fc = sscha.harmonic.compute_fcs()
            datname = "band_harmonic.dat"
            yamlname = "band_harmonic.yaml"
        else:
            fc = phonon.force_constants
            datname = "band.dat"
            yamlname = "band.yaml"

        unitcell = phonon.unitcell
        supercell = phonon.supercell
        supercell_matrix = phonon.get_supercell_matrix()

        phonon = Phonopy(
            unitcell,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive,
        )
        if os.path.exists("BORN"):
            nac_params = parse_BORN(phonon.primitive)
            phonon = Phonopy(
                unitcell,
                supercell_matrix=supercell_matrix,
                primitive_matrix=primitive,
                nac_params=nac_params,
            )
        phonon.force_constants = fc

        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=101)
        phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
        band_dict = phonon.get_band_structure_dict()
        phonon.write_yaml_band_structure(filename=yamlname)
        f = open(datname, "w")
        for d1, freq in zip(band_dict["distances"], band_dict["frequencies"]):
            for d2, f2 in zip(d1, freq):
                print(d2, *f2, file=f)
        f.close()
