"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal
import sys

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.common_args import (
    create_advanced_sscha_parser,
    create_go_parser,
    create_gsfe_parser,
    create_sscha_parser,
    create_structure_parser,
)
from pypolymlp.api.func_sscha import run_main_sscha
from pypolymlp.core.utils import print_credit

from .lammps_args import create_lammps_parser


def _parse_args_pypolymlp_sscha(args=None):
    """Parse options."""
    lammps_parser = create_lammps_parser()
    st_parser = create_structure_parser()
    sscha_parser = create_sscha_parser()
    go_parser = create_go_parser(default_gtol=0.01)
    gsfe_parser = create_gsfe_parser()
    advanced_sscha_parser = create_advanced_sscha_parser()

    parser = argparse.ArgumentParser(
        description="SSCHA calculations using Lammps",
        parents=[
            lammps_parser,
            st_parser,
            sscha_parser,
            advanced_sscha_parser,
            go_parser,
            gsfe_parser,
        ],
    )
    args = parser.parse_args(args)
    return args


def run():
    """Run command."""
    print_credit()
    np.set_printoptions(legacy="1.21")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = _parse_args_pypolymlp_sscha(sys.argv[1:])

    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_lammps(
        elements=args.elements,
        pot=args.pot,
        style=args.style,
        style_command=args.style_command,
        coeff_command=args.coeff_command,
    )
    run_main_sscha(args, polymlp)
