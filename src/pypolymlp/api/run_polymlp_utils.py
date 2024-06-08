#!/usr/bin/env python
import argparse
import signal

from pypolymlp.mlp_opt.optimal import find_optimal_mlps
from pypolymlp.utils.atomic_energies.atomic_energies import (
    get_atomic_energies_polymlp_in,
)
from pypolymlp.utils.count_time import PolymlpCost
from pypolymlp.utils.dataset.auto_divide import auto_divide
from pypolymlp.utils.vasp_utils import print_poscar, write_poscar_file
from pypolymlp.utils.vasprun_compress import convert


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vasprun_compress",
        nargs="*",
        type=str,
        default=None,
        help="Compression of vasprun.xml files",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")

    parser.add_argument(
        "--auto_dataset",
        nargs="*",
        type=str,
        default=None,
        help="Automatic dataset division using " + "vasprun.xml files",
    )

    parser.add_argument(
        "--atomic_energy_elements",
        nargs="*",
        type=str,
        default=None,
        help="Elements for getting atomic energies.",
    )
    parser.add_argument(
        "--atomic_energy_formula",
        type=str,
        default=None,
        help="Compound for getting atomic energies.",
    )
    parser.add_argument(
        "--atomic_energy_functional",
        type=str,
        default="PBE",
        help="Exc functional for getting atomic energies.",
    )

    """Calculation of computational costs"""
    parser.add_argument(
        "--calc_cost",
        action="store_true",
        help="Calculation of computational costs.",
    )
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="*",
        type=str,
        default=None,
        help="directory paths",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.lammps",
        help="polymlp file",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=[4, 4, 4],
        help="supercell size",
    )
    parser.add_argument("--n_calc", type=int, default=20, help="number of calculations")

    """Pareto optimal search"""
    parser.add_argument(
        "--find_optimal",
        nargs="*",
        type=str,
        default=None,
        help="Find optimal MLPs using a set of MLPs. "
        + "Directories for the set of MLPs.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Identification key for the dataset " + "in finding optimal MLPs",
    )

    """Spglib utilities"""
    parser.add_argument("-p", "--poscar", type=str, help="poscar file name")
    parser.add_argument(
        "--symprec",
        type=float,
        default=1e-4,
        help="numerical precision for finding symmetry",
    )
    parser.add_argument("--refine_cell", action="store_true", help="refine cell")
    parser.add_argument("--space_group", action="store_true", help="get space group")

    args = parser.parse_args()

    if args.vasprun_compress is not None:
        if args.n_jobs == 1:
            for vasp in args.vasprun_compress:
                convert(vasp)
        else:
            from joblib import Parallel, delayed

            _ = Parallel(n_jobs=args.n_jobs)(
                delayed(convert)(vasp) for vasp in args.vasprun_compress
            )

    elif args.auto_dataset is not None:
        auto_divide(args.auto_dataset)

    elif (
        args.atomic_energy_elements is not None
        or args.atomic_energy_formula is not None
    ):
        get_atomic_energies_polymlp_in(
            elements=args.atomic_energy_elements,
            formula=args.atomic_energy_formula,
            functional=args.atomic_energy_functional,
        )

    elif args.calc_cost:
        pycost = PolymlpCost(
            pot_path=args.dirs,
            pot=args.pot,
            poscar=args.poscar,
            supercell=args.supercell,
        )
        pycost.run(n_calc=args.n_calc)
    elif args.find_optimal is not None:
        find_optimal_mlps(args.find_optimal, args.key)

    elif args.refine_cell or args.space_group:
        from pypolymlp.utils.spglib_utils import SymCell

        sc = SymCell(args.poscar, symprec=args.symprec)
        if args.refine_cell:
            st_dict = sc.refine_cell()
            print_poscar(st_dict)
            write_poscar_file(st_dict)
        if args.space_group:
            print(" space_group = ", sc.get_spacegroup())
