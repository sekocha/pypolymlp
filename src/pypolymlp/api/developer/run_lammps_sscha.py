"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal

import numpy as np

from pypolymlp.api.common_args import (
    create_advanced_sscha_parser,
    create_go_parser,
    create_sscha_parser,
    create_structure_parser,
)
from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.api.run_polymlp_sscha import run_main_sscha
from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps
from pypolymlp.core.utils import print_credit

from .lammps_args import create_lammps_parser


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    lammps_parser = create_lammps_parser()
    st_parser = create_structure_parser()
    sscha_parser = create_sscha_parser()

    go_parser = create_go_parser(default_gtol=0.01)
    advanced_sscha_parser = create_advanced_sscha_parser()
    parser = argparse.ArgumentParser(
        description="SSCHA calculations using PolyMLP",
        parents=[
            lammps_parser,
            st_parser,
            sscha_parser,
            advanced_sscha_parser,
            go_parser,
        ],
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    prop = PropertiesLammps(
        elements=args.elements,
        pot=args.pot,
        style=args.style,
        style_command=args.style_command,
        coeff_command=args.coeff_command,
        verbose=False,
    )

    sscha = PypolymlpSSCHA(verbose=True)
    if args.pot is not None:
        sscha.set_polymlp(properties=prop)
    sscha._pot = args.pot

    run_main_sscha(args, sscha)
