#!/usr/bin/env python
import argparse
import signal

from pypolymlp.str_gen.strgen import run_strgen

# from pypolymlp.str_gen.prototypes_selection import (
#    prototype_selection_element,
#    prototype_selection_alloy,
#    check_compositions,
# )


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()

    """
    parser.add_argument('--prototypes',
                        action='store_true',
                        help='Prototype structure generation')
    parser.add_argument('--n_types',
                        type=int,
                        default=None,
                        help='Number of atom types (n_types = 1,2,3)')
    parser.add_argument('--comp',
                        type=float,
                        nargs='*',
                        default=None,
                        help='Composition')
    parser.add_argument('--noscreen',
                        action='store_false',
                        help='All nonequivalent prototypes are generated.')
    """

    parser.add_argument(
        "--random", action="store_true", help="Random structure generation"
    )
    parser.add_argument(
        "-p",
        "--poscars",
        type=str,
        nargs="*",
        help="Initial structures in POSCAR format",
    )
    parser.add_argument(
        "--max_natom",
        type=int,
        default=150,
        help="Maximum number of atoms in structures",
    )
    parser.add_argument(
        "--n_str",
        type=int,
        default=None,
        help=("Number of sample structures. " "(for --random and --random_phonon)"),
    )
    parser.add_argument(
        "--max_disp",
        type=float,
        default=1.5,
        help="Maximum random number for generating " "atomic displacements",
    )

    parser.add_argument(
        "--low_density",
        type=int,
        default=None,
        help="Number of structures for low density mode.",
    )
    parser.add_argument(
        "--high_density",
        type=int,
        default=None,
        help="Number of structures for high density mode.",
    )
    parser.add_argument(
        "--density_mode_disp",
        type=float,
        default=0.2,
        help="Maximum random number for generating atomic "
        "displacements in low and high density modes",
    )

    parser.add_argument(
        "--random_phonon",
        action="store_true",
        help="Random displacement generation",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=[2, 2, 2],
        help=("Supercell size for random displacement" " generation"),
    )
    parser.add_argument(
        "--disp",
        type=float,
        default=0.03,
        help="Random displacement in Angstrom",
    )

    args = parser.parse_args()

    """
    if args.prototypes:
        if args.n_types is None:
            raise ValueError('error: --n_types is required for --prototype.')

        if args.n_types == 1:
            prototype_selection_element(screen=args.noscreen)
        else:
            target = 'alloy' # 'ionic' must be hidden
            comp = check_compositions(args.comp, args.n_types)
            print(' composition =',  comp)
            prototype_selection_alloy(args.n_types,
                                      target=target,
                                      screen=args.noscreen,
                                      comp=comp)
    """
    if args.random:
        if args.poscars is None:
            raise ValueError("error: -p/--poscars is required for --random.")
        run_strgen(args)

    if args.random_phonon:
        if args.poscars is None:
            raise ValueError(("error: -p/--poscars is required" "for --random_phonon."))

        from pypolymlp.str_gen.strgen_phonon import run_strgen_phonon

        for poscar in args.poscars:
            run_strgen_phonon(
                poscar,
                supercell_size=args.supercell,
                n_samples=args.n_str,
                displacements=args.disp,
            )
