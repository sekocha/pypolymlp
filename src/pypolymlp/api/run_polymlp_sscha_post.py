"""Command lines for post SSCHA calculations."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha_post import PolymlpSSCHAPost


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

    sscha = PolymlpSSCHAPost(verbose=True)
    if args.properties:
        sscha.compute_thermodynamic_properties(
            args.yaml, filename="sscha_properties.yaml"
        )
