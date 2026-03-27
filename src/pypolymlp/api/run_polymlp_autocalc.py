"""Command lines for systematically calculating properites."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.core.utils import print_credit


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pot", nargs="*", type=str, default=None, help="polymlp file")

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    polymlp = PypolymlpAutoCalc(pot=args.pot, verbose=True)
    polymlp.run_prototypes()
    polymlp.save_prototypes()
