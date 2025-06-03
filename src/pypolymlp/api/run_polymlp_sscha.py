"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.core.utils import print_credit


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
        help="polymlp files",
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
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter")
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
    parser.add_argument(
        "--born_vasprun",
        type=str,
        default=None,
        help="vasprun.xml file for parsing born effective charges",
    )
    parser.add_argument(
        "--cutoff_fc2",
        type=float,
        default=None,
        help="Cutoff radius for effective force constants.",
    )
    parser.add_argument(
        "--use_temporal_cutoff",
        action="store_true",
        help="Use an algorithm temporarily using cutoff radius.",
    )

    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    print_credit()
    sscha = PypolymlpSSCHA(verbose=True)
    if args.poscar is not None:
        sscha.load_poscar(args.poscar, np.diag(args.supercell))
    elif args.yaml is not None:
        sscha.load_restart(yaml=args.yaml, parse_fc2=True)

    if args.pot is not None:
        sscha.set_polymlp(args.pot)
    if args.born_vasprun is not None:
        sscha.set_nac_params(args.born_vasprun)

    if args.n_samples is None:
        n_samples_init = None
        n_samples_final = None
    else:
        n_samples_init, n_samples_final = args.n_samples

    sscha.run(
        temp=args.temp,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        temp_step=args.temp_step,
        ascending_temp=args.ascending_temp,
        n_samples_init=n_samples_init,
        n_samples_final=n_samples_final,
        tol=args.tol,
        max_iter=args.max_iter,
        mixing=args.mixing,
        mesh=args.mesh,
        init_fc_algorithm=args.init,
        init_fc_file=args.init_file,
        cutoff_radius=args.cutoff_fc2,
        use_temporal_cutoff=args.use_temporal_cutoff,
    )
