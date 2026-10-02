"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.core.utils import print_credit

from .common_args import (
    create_advanced_sscha_parser,
    create_go_parser,
    create_polymlp_parser,
    create_sscha_parser,
    create_structure_parser,
)
from .func_sscha import run_main_sscha


def run():
    """Run command line."""

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    polymlp_parser = create_polymlp_parser()
    st_parser = create_structure_parser()
    sscha_parser = create_sscha_parser()

    go_parser = create_go_parser(default_gtol=0.01)
    advanced_sscha_parser = create_advanced_sscha_parser()
    parser = argparse.ArgumentParser(
        description="SSCHA calculations using PolyMLP",
        parents=[
            polymlp_parser,
            st_parser,
            sscha_parser,
            advanced_sscha_parser,
            go_parser,
        ],
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()

    polymlp = PypolymlpCalcProperties(verbose=True)
    if args.pot is not None:
        polymlp.set_polymlp(pot=args.pot)
    else:
        polymlp.load_sscha_restart(yaml=args.yaml)

    run_main_sscha(args, polymlp)
