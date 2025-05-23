"""Command lines for calculating thermodynamic properties."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_thermodynamics import PypolymlpThermodynamics


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sscha",
        nargs="*",
        type=str,
        required=True,
        help="yaml files of SSCHA results",
    )
    parser.add_argument(
        "--electron",
        nargs="*",
        type=str,
        default=None,
        help="yaml files of electronic contribution",
    )
    parser.add_argument(
        "--ti",
        nargs="*",
        type=str,
        default=None,
        help="yaml files of thermodynamic integration results",
    )
    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    thermo = PypolymlpThermodynamics(args.sscha, args.electron, args.ti, verbose=True)
    thermo.run()
    thermo.save_sscha(filename="polymlp_thermodynamics_sscha.yaml")
    if args.electron is not None or args.ti is not None:
        thermo.save_total(filename="polymlp_thermodynamics_total.yaml")
