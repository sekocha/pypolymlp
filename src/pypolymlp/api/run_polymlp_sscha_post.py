"""Command lines for post SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha_post import PypolymlpSSCHAPost
from pypolymlp.core.utils import print_credit


def run():
    """Command lines for post SSCHA calculations.

    Examples
    --------
    # Calculation of energies and forces for structures generated from a converged FC2
    pypolymlp-sscha-post --distribution --yaml sscha/300/sscha_results.yaml
                         --fc2 sscha/300/fc2.hdf5 --n_samples 20 --pot polymlp.lammps

    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Calculate properties of structures sampled from FC2 density matrix.",
    )
    parser.add_argument(
        "--yaml",
        nargs="*",
        type=str,
        default=None,
        help="sscha_results.yaml files",
    )
    parser.add_argument(
        "--fc2",
        type=str,
        default="fc2.hdf5",
        help="fc2.hdf5 file to be parsed.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help="polymlp files",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=100,
        help="Number of sample supercells",
    )
    parser.add_argument(
        "--save_poscars",
        action="store_true",
        help="Save POSCAR files of distribution",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    sscha = PypolymlpSSCHAPost(verbose=True)
    if args.distribution:
        sscha.init_structure_distribution(
            yamlfile=args.yaml[0],
            fc2file=args.fc2,
            pot=args.pot,
        )
        sscha.run_structure_distribution(n_samples=args.n_samples)
        sscha.save_structure_distribution(path=".", save_poscars=args.save_poscars)
