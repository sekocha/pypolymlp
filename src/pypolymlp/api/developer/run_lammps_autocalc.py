"""Command lines for systematically calculating properites."""

import argparse
import os
import signal

import numpy as np

from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps
from pypolymlp.core.utils import print_credit


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs="*",
        type=str,
        required=True,
        help="Element list",
    )
    parser.add_argument(
        "--pot", type=str, default="polymlp.yaml", help="Potential file"
    )
    parser.add_argument("--style", type=str, default="polymlp", help="Potential style")
    parser.add_argument(
        "--style_command",
        type=str,
        default="pair_style",
        help="Potential style header",
    )
    parser.add_argument(
        "--coeff_command",
        type=str,
        default="pair_coeff",
        help="Potential coeff header",
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
