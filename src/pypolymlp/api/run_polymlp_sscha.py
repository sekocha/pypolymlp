"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal
import sys

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.core.utils import print_credit

from .common_args import (
    create_advanced_sscha_parser,
    create_go_parser,
    create_gsfe_parser,
    create_polymlp_parser,
    create_sscha_parser,
    create_structure_parser,
)
from .func_sscha import run_main_sscha


def _parse_args_pypolymlp_sscha(args=None):
    """Parse options."""
    polymlp_parser = create_polymlp_parser()
    st_parser = create_structure_parser()
    sscha_parser = create_sscha_parser()
    go_parser = create_go_parser(default_gtol=0.01)
    gsfe_parser = create_gsfe_parser()
    advanced_sscha_parser = create_advanced_sscha_parser()

    parser = argparse.ArgumentParser(
        description="SSCHA calculations using PolyMLP",
        parents=[
            polymlp_parser,
            st_parser,
            sscha_parser,
            advanced_sscha_parser,
            go_parser,
            gsfe_parser,
        ],
    )
    args = parser.parse_args(args)
    return args


def run():
    """Run command line."""

    print_credit()
    np.set_printoptions(legacy="1.21")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = _parse_args_pypolymlp_sscha(sys.argv[1:])

    polymlp = PypolymlpCalcProperties(verbose=True)
    if args.pot is not None:
        polymlp.set_polymlp(pot=args.pot)
    else:
        polymlp.load_sscha_restart(yaml=args.yaml)

    run_main_sscha(args, polymlp)
