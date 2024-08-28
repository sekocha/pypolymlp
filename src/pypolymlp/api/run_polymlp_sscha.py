"""Function for performing SSCHA calculations by command line."""

import argparse
import signal

import numpy as np

from pypolymlp.calculator.sscha.run_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_utils import (
    Restart,
    n_samples_setting,
    print_parameters,
    print_structure,
    temperature_setting,
)
from pypolymlp.core.interface_vasp import Poscar


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default=None,
        help="poscar file (unit cell)",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="sscha_results.yaml file for parsing " "unitcell and supercell size.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help="polymlp.lammps file",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="q-mesh for phonon calculation",
    )
    parser.add_argument(
        "-t", "--temp", type=float, default=None, help="Temperature (K)"
    )
    parser.add_argument(
        "-t_min",
        "--temp_min",
        type=float,
        default=100,
        help="Lowest temperature (K)",
    )
    parser.add_argument(
        "-t_max",
        "--temp_max",
        type=float,
        default=2000,
        help="Highest temperature (K)",
    )
    parser.add_argument(
        "-t_step",
        "--temp_step",
        type=float,
        default=100,
        help="Temperature interval (K)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance parameter for FC convergence",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        nargs=2,
        default=None,
        help="Number of steps used in " "iterations and the last iteration",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--ascending_temp",
        action="store_true",
        help="Use ascending order of temperatures",
    )
    parser.add_argument(
        "--init",
        choices=["harmonic", "const", "random", "file"],
        default="harmonic",
        help="Initial FCs",
    )
    parser.add_argument(
        "--init_file",
        default=None,
        help="Location of fc2.hdf5 for initial FCs",
    )
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter")
    args = parser.parse_args()

    if args.poscar is not None:
        unitcell = Poscar(args.poscar).structure
        supercell_matrix = np.diag(args.supercell)
    elif args.yaml is not None:
        res = Restart(args.yaml)
        unitcell = res.unitcell
        supercell_matrix = res.supercell_matrix
        if args.pot is None:
            args.pot = res.mlp

    n_atom = len(unitcell.elements) * np.linalg.det(supercell_matrix)
    args = temperature_setting(args)
    args = n_samples_setting(args, n_atom)

    print_parameters(supercell_matrix, args)
    print_structure(unitcell)

    run_sscha(unitcell, supercell_matrix, args, pot=args.pot)
