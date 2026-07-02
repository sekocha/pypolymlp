"""Command lines for calculating properites using lammps."""

import argparse
import signal

import numpy as np

from pypolymlp.api.common_args import (
    create_fc_parser,
    create_go_parser,
    create_mode_parser,
    create_phonon_parser,
    create_structure_parser,
)
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.run_polymlp_calc import check_variables, run_calculations
from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps
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

    parser = argparse.ArgumentParser(
        description="Calculations using interatomic potentials in Lammps",
        parents=[
            mode_parser,
            lammps_parser,
            st_parser,
            go_parser,
            phonon_parser,
            fc_parser,
        ],
    )

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()
    args = check_variables(args)

    prop = PropertiesLammps(
        elements=args.elements,
        pot=args.pot,
        style=args.style,
        style_command=args.style_command,
        coeff_command=args.coeff_command,
        verbose=False,
    )
    polymlp = PypolymlpCalc(properties=prop, verbose=True)
    run_calculations(args, polymlp, calc_features=False)
