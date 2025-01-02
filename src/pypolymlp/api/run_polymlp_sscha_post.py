"""Command lines for post SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.calculator.sscha.sscha_properties import SSCHAProperties


def run():
    """Command lines for post SSCHA calculations."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--properties",
        action="store_true",
        help="Calculate thermodynamic properties.",
    )
    parser.add_argument(
        "--yaml",
        nargs="*",
        type=str,
        default=None,
        help="sscha_results.yaml files",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")

    if args.properties:
        sscha = SSCHAProperties(args.yaml, verbose=True)
        sscha.run()
        sscha.save_properties(filename="sscha_properties.yaml")
