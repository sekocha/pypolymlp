"""Command lines for systematically calculating properites."""

import argparse
import os
import signal

import numpy as np

from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps
from pypolymlp.core.utils import print_credit

from .lammps_args import create_lammps_parser


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    lammps_parser = create_lammps_parser()
    parser = argparse.ArgumentParser(
        description="Automated calculations using interatomic potentials in Lammps",
        parents=[lammps_parser],
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
    polymlp = PypolymlpAutoCalc(properties=prop, verbose=True)
    polymlp.run_prototypes()
    polymlp.save_prototypes()
    os._exit(0)
