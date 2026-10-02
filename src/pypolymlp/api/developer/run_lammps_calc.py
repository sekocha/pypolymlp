"""Command lines for calculating properites using lammps."""

import argparse
import signal

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.common_args import (
    create_fc_parser,
    create_go_parser,
    create_gsfe_parser,
    create_mode_parser,
    create_phonon_parser,
    create_structure_parser,
)
from pypolymlp.api.func_calc import run_calculations
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.run_polymlp_calc import check_variables
from pypolymlp.core.utils import print_credit

from .lammps_args import create_lammps_parser


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    mode_parser, _ = create_mode_parser()
    lammps_parser = create_lammps_parser()
    st_parser = create_structure_parser(multiple=True, enable_yaml=True)
    fc_parser = create_fc_parser()
    go_parser = create_go_parser()
    phonon_parser = create_phonon_parser()
    gsfe_parser = create_gsfe_parser()

    parser = argparse.ArgumentParser(
        description="Calculations using interatomic potentials in Lammps",
        parents=[
            mode_parser,
            lammps_parser,
            st_parser,
            go_parser,
            phonon_parser,
            gsfe_parser,
            fc_parser,
        ],
    )

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()
    args = check_variables(args)

    polymlp = PypolymlpCalcProperties(verbose=True)
    prop = polymlp.set_lammps(
        elements=args.elements,
        pot=args.pot,
        style=args.style,
        style_command=args.style_command,
        coeff_command=args.coeff_command,
    )
    polymlp = PypolymlpCalc(properties=prop, verbose=True)
    run_calculations(args, polymlp, calc_features=False)
