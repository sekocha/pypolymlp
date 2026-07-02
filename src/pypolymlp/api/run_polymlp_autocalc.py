"""Command lines for systematically calculating properites."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.core.utils import print_credit

from .common_args import create_polymlp_parser


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    polymlp_parser = create_polymlp_parser()
    parser = argparse.ArgumentParser(
        description="Automated calculations using PolyMLP",
        parents=[polymlp_parser],
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    polymlp = PypolymlpAutoCalc(pot=args.pot, verbose=True)
    polymlp.run_prototypes()
    polymlp.save_prototypes()
