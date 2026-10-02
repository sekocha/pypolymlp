"""Command lines for calculating properites using polynomial MLP."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.utils import print_credit

from .common_args import (
    create_fc_parser,
    create_go_parser,
    create_mode_parser,
    create_phonon_parser,
    create_polymlp_parser,
    create_structure_parser,
)
from .func_calc import run_calculations


def check_variables(args):
    """Check variables."""
    if args.poscar is None and args.poscars is not None:
        args.poscar = args.poscars
    if args.poscars is None and args.poscar is not None:
        args.poscars = args.poscar
    return args


def run():
    """Main code for command line."""

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    mode_parser, mode_group = create_mode_parser()
    mode_group.add_argument(
        "--features", action="store_true", help="Mode: Feature calculation"
    )

    polymlp_parser = create_polymlp_parser()
    st_parser = create_structure_parser(multiple=True, enable_yaml=True)
    fc_parser = create_fc_parser()
    go_parser = create_go_parser()
    phonon_parser = create_phonon_parser()

    parser = argparse.ArgumentParser(
        description="Calculations using PolyMLP",
        parents=[
            mode_parser,
            polymlp_parser,
            st_parser,
            go_parser,
            phonon_parser,
            fc_parser,
        ],
    )

    feature_group = parser.add_argument_group(
        "Features", "Options for structural feature calculation"
    )
    feature_group.add_argument(
        "-i",
        "--infile",
        type=str,
        default=None,
        help="Input file name",
    )
    feature_group.add_argument(
        "--precision",
        action="store_true",
        help="Mode: MLP precision calculation. This uses only features",
    )

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    args = check_variables(args)
    if args.pot is None and args.infile is None:
        raise RuntimeError("Input parameters not found.")
    require_mlp = True if args.pot is not None else False

    polymlp = PypolymlpCalc(pot=args.pot, verbose=True, require_mlp=require_mlp)
    run_calculations(args, polymlp)
