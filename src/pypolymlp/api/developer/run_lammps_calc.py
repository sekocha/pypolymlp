"""Command lines for calculating properites using lammps."""

import argparse
import signal
import sys

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.common_args import (
    check_poscar_variables,
    create_fc_parser,
    create_go_parser,
    create_gsfe_parser,
    create_mode_parser,
    create_phonon_parser,
    create_structure_parser,
)
from pypolymlp.api.func_calc import run_calculations
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.utils import print_credit

from .lammps_args import create_lammps_parser


def _parse_args_lammps_calc(args=None):
    """Parse args."""
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
    args = parser.parse_args(args)
    args = check_poscar_variables(args)
    return args


def run():
    """Run command."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print_credit()
    np.set_printoptions(legacy="1.21")
    args = _parse_args_lammps_calc(sys.argv[1:])

    polymlp = PypolymlpCalcProperties(verbose=True)
    prop = polymlp.set_lammps(
        elements=args.elements,
        pot=args.pot,
        style=args.style,
        style_command=args.style_command,
        coeff_command=args.coeff_command,
    )
    calc = PypolymlpCalc(properties=prop, verbose=True)
    run_calculations(args, calc, calc_features=False)
