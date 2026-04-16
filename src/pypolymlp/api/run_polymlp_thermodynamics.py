"""Command lines for calculating thermodynamic properties."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_thermodynamics import (
    PypolymlpThermodynamics,
    PypolymlpTransition,
)
from pypolymlp.core.utils import print_credit


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sscha",
        nargs="*",
        type=str,
        default=None,
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
    parser.add_argument(
        "--electron_phonon",
        nargs="*",
        type=str,
        default=None,
        help="yaml files of electron-phonon contribution",
    )
    parser.add_argument(
        "--boundary",
        nargs=2,
        type=str,
        help="Find phase boundary.",
    )
    parser.add_argument(
        "--ref_fc2",
        nargs="*",
        type=str,
        default=None,
        help="FC2 files used for reference harmonic state.",
    )

    args = parser.parse_args()

    np.set_printoptions(legacy="1.21")
    print_credit()

    if args.boundary:
        transition = PypolymlpTransition(
            args.boundary[0],
            args.boundary[1],
            verbose=True,
        )
        tc = transition.find_phase_transition()
        pd = transition.compute_phase_boundary()
        np.set_printoptions(suppress=True)
        print("Tc at 0 GPa:", tc, flush=True)
        print("Phase boundary:", flush=True)
        print(pd)
    else:
        if args.sscha is None:
            raise RuntimeError("yaml files of SSCHA results are required.")

        thermo = PypolymlpThermodynamics(
            args.sscha,
            args.electron,
            args.ti,
            args.electron_phonon,
            ref_fc2=args.ref_fc2,
            verbose=True,
        )
        thermo.run()
        thermo.save(path="polymlp_thermodynamics")
