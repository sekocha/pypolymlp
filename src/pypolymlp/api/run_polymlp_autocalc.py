"""Command lines for systematically calculating properites."""

import argparse
import signal
import sys

import numpy as np

from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.core.utils import print_credit

from .common_args import create_polymlp_parser


def _parse_args_pypolymlp_autocalc(args=None):
    """Parse options."""
    polymlp_parser = create_polymlp_parser()
    parser = argparse.ArgumentParser(
        description="Automated calculations using PolyMLP",
        parents=[polymlp_parser],
    )
    args = parser.parse_args(args)
    return args


def run():

    print_credit()
    np.set_printoptions(legacy="1.21")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = _parse_args_pypolymlp_autocalc(sys.argv[1:])

    polymlp = PypolymlpAutoCalc(pot=args.pot, verbose=True)
    polymlp.run_prototypes()
    polymlp.save_prototypes()
