"""Command lines for using utilities."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_utils import PypolymlpUtils
from pypolymlp.core.utils import print_credit
from pypolymlp.utils.atomic_energies.atomic_energies import (
    get_atomic_energies_polymlp_in,
)


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
        "--electron_vasprun",
        nargs="*",
        type=str,
        default=None,
        help="Parse vasprun.xml files to get electronic properties.",
    )
    parser.add_argument(
        "--temp_max",
        type=float,
        default=1000,
        help="Maximum temperature (K).",
    )
    parser.add_argument(
        "--temp_step",
        type=float,
        default=10,
        help="Temperature interval (K).",
    )

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
        default="polymlp.yaml",
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
    print_credit()

    np.set_printoptions(legacy="1.21")
    polymlp = PypolymlpUtils(verbose=True)

    if args.electron_vasprun is not None:
        polymlp.compute_electron_properties_from_vaspruns(
            args.electron_vasprun,
            temp_max=args.temp_max,
            temp_step=args.temp_step,
            n_jobs=args.n_jobs,
        )
    elif args.vasprun_compress is not None:
        polymlp.compress_vaspruns(args.vasprun_compress, n_jobs=args.n_jobs)
    elif args.calc_cost:
        polymlp.estimate_polymlp_comp_cost(
            pot=args.pot,
            path_pot=args.dirs,
            poscar=args.poscar,
            supercell=args.supercell,
            n_calc=args.n_calc,
        )
    elif args.find_optimal is not None:
        polymlp.find_optimal_mlps(args.find_optimal, args.key)

    elif args.auto_dataset is not None:
        polymlp.divide_dataset(args.auto_dataset)

    elif args.refine_cell or args.space_group:
        polymlp.init_symmetry(poscar=args.poscar, symprec=args.symprec)
        if args.refine_cell:
            structure = polymlp.refine_cell()
            polymlp.print_poscar(structure)
            polymlp.write_poscar_file(structure, filename="poscar_pypolymlp")
        if args.space_group:
            print(" space_group = ", polymlp.get_spacegroup(), flush=True)
    elif (
        args.atomic_energy_elements is not None
        or args.atomic_energy_formula is not None
    ):
        get_atomic_energies_polymlp_in(
            elements=args.atomic_energy_elements,
            formula=args.atomic_energy_formula,
            functional=args.atomic_energy_functional,
        )
