"""Command lines arguments."""

import argparse


def create_polymlp_parser():
    """Create parser for polymlp arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    pot_group = parser.add_argument_group("Polymlp", "Options for setting polymlp")
    pot_group.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help=(
            "PolyMLP file. When multiple files are specified, "
            "the corresponding hybrid model will be used."
        ),
    )
    return parser


def create_structure_parser(
    multiple: bool = False,
    enable_yaml: bool = False,
):
    """Create parser for common arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    st_group = parser.add_argument_group(
        "Structure", "Options for setting input structure(s)"
    )
    st_group.add_argument("-p", "--poscar", type=str, default=None, help="POSCAR file")
    if multiple:
        st_group.add_argument(
            "--poscars", nargs="*", type=str, default=None, help="POSCAR files"
        )
        st_group.add_argument(
            "--vaspruns",
            nargs="*",
            type=str,
            default=None,
            help="vasprun.xml files",
        )
    if enable_yaml:
        st_group.add_argument(
            "--str_yaml", type=str, default=None, help="polymlp_str.yaml file"
        )
    st_group.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=(2, 2, 2),
        help="Supercell size (diagonal components)",
    )
    return parser


def create_mode_parser():
    """Create parser for specifying calculation."""
    parser = argparse.ArgumentParser(add_help=False)
    mode_group = parser.add_argument_group(
        "Calculation mode", "Options for specifying calculation mode"
    )
    mode_group.add_argument(
        "--properties",
        action="store_true",
        help="Mode: Property calculation",
    )
    mode_group.add_argument(
        "--force_constants",
        action="store_true",
        help="Mode: Force constant calculation",
    )
    mode_group.add_argument(
        "--phonon", action="store_true", help="Mode: Phonon calculation"
    )
    mode_group.add_argument("--eos", action="store_true", help="Mode: EOS calculation")
    mode_group.add_argument(
        "--elastic", action="store_true", help="Mode: Elastic constant calculation"
    )
    mode_group.add_argument(
        "--geometry_optimization",
        action="store_true",
        help="Geometry optimization is performed for initial structure.",
    )
    return parser, mode_group


def create_fc_parser():
    """Create parser for force constant calculation."""
    parser = argparse.ArgumentParser(add_help=False)
    fc_group = parser.add_argument_group(
        "Force constants", "Options for calculationg force constants"
    )
    fc_group.add_argument(
        "--fc_n_samples",
        type=int,
        default=None,
        help="Number of random displacement samples",
    )
    fc_group.add_argument(
        "--disp",
        type=float,
        default=0.01,
        help="Displacement (in Angstrom)",
    )
    fc_group.add_argument(
        "--is_plusminus",
        action="store_true",
        help="Plus-minus displacements will be generated.",
    )
    fc_group.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size for FC solver.",
    )
    fc_group.add_argument(
        "--cutoff_fc2",
        type=float,
        default=None,
        help="Cutoff radius for FC2 to set zero elements.",
    )
    fc_group.add_argument(
        "--cutoff_fc3",
        type=float,
        default=None,
        help="Cutoff radius for FC3 to set zero elements.",
    )
    fc_group.add_argument(
        "--cutoff_fc4",
        type=float,
        default=None,
        help="Cutoff radius for FC4 to set zero elements.",
    )
    fc_group.add_argument(
        "--fc_orders",
        nargs="*",
        type=int,
        default=(2, 3),
        help="FC orders.",
    )
    fc_group.add_argument(
        "--use_gradient_solver",
        action="store_true",
        help="Use gradient-based solver in force-constant estimation.",
    )
    fc_group.add_argument("--run_ltc", action="store_true", help="Run LTC calculations")
    fc_group.add_argument(
        "--ltc_mesh",
        type=int,
        nargs=3,
        default=[19, 19, 19],
        help="k-mesh used for phono3py calculation",
    )
    return parser


def create_go_parser():
    """Create parser for geometry optimization."""
    parser = argparse.ArgumentParser(add_help=False)
    go_group = parser.add_argument_group(
        "Geometry optimization", "Options for geometry optimization"
    )
    go_group.add_argument(
        "--pressure",
        type=float,
        default=0.0,
        help="Pressure (in GPa)",
    )
    go_group.add_argument(
        "--no_symmetry",
        action="store_true",
        help="Ignore symmetric properties in geometry optimization",
    )
    go_group.add_argument(
        "--fix_cell",
        action="store_true",
        help="Fix cell shape and volume in geometry optimization",
    )
    go_group.add_argument(
        "--fix_volume",
        action="store_true",
        help="Fix cell volume in geometry optimization",
    )
    go_group.add_argument(
        "--fix_atom",
        action="store_true",
        help="Fix atomic positions in geometry optimization",
    )
    go_group.add_argument(
        "--method",
        type=str,
        choices=["BFGS", "CG", "L-BFGS-B", "SLSQP"],
        default="BFGS",
        help="Algorithm for geometry optimization",
    )
    return parser


def create_phonon_parser():
    """Create parser for phonon calculation."""
    parser = argparse.ArgumentParser(add_help=False)
    ph_group = parser.add_argument_group("Phonon", "Options for phonon calculation")
    ph_group.add_argument(
        "--ph_mesh",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="k-mesh used for phonon calculation",
    )
    ph_group.add_argument("--ph_tmin", type=float, default=0, help="Temperature (min)")
    ph_group.add_argument(
        "--ph_tmax", type=float, default=1000, help="Temperature (max)"
    )
    ph_group.add_argument(
        "--ph_tstep", type=float, default=10, help="Temperature (step)"
    )
    ph_group.add_argument("--ph_pdos", action="store_true", help="Compute phonon PDOS")
    return parser
