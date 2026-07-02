"""Command lines arguments."""

import argparse


def create_lammps_parser():
    """Create parser for lammps arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    pot_group = parser.add_argument_group(
        "Lammps", "Options for setting Lammps potential"
    )
    pot_group.add_argument(
        "--elements",
        nargs="*",
        type=str,
        required=True,
        help="Element list",
    )
    pot_group.add_argument(
        "--pot", type=str, default="polymlp.yaml", help="Potential file"
    )
    pot_group.add_argument(
        "--style", type=str, default="polymlp", help="Potential style"
    )
    pot_group.add_argument(
        "--style_command",
        type=str,
        default="pair_style",
        help="Potential style header",
    )
    pot_group.add_argument(
        "--coeff_command",
        type=str,
        default="pair_coeff",
        help="Potential coeff header",
    )
    return parser
