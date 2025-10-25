"""Command lines for calculating properites using polynomial MLP."""

import argparse
import signal
import time

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.utils import precision, print_credit


def check_variables(args):
    """Check variables."""
    if args.poscar is None and args.poscars is not None:
        args.poscar = args.poscars
    if args.poscars is None and args.poscar is not None:
        args.poscars = args.poscar
    return args


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--properties",
        action="store_true",
        help="Mode: Property calculation",
    )
    parser.add_argument(
        "--force_constants",
        action="store_true",
        help="Mode: Force constant calculation",
    )
    parser.add_argument(
        "--phonon", action="store_true", help="Mode: Phonon calculation"
    )
    parser.add_argument(
        "--features", action="store_true", help="Mode: Feature calculation"
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help="Mode: MLP precision calculation. This uses only features",
    )
    parser.add_argument("--eos", action="store_true", help="Mode: EOS calculation")
    parser.add_argument(
        "--elastic", action="store_true", help="Mode: Elastic constant calculation"
    )

    parser.add_argument("--pot", nargs="*", type=str, default=None, help="polymlp file")
    parser.add_argument(
        "--poscars", nargs="*", type=str, default=None, help="poscar files"
    )
    parser.add_argument(
        "--vaspruns",
        nargs="*",
        type=str,
        default=None,
        help="vasprun files",
    )
    parser.add_argument(
        "--phono3py_yaml", type=str, default=None, help="phono3py.yaml file"
    )
    parser.add_argument(
        "--phono3py_yaml_structure_ids",
        nargs=2,
        type=int,
        default=None,
        help="Structure range in phono3py.yaml file",
    )

    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--str_yaml", type=str, default=None, help="polymlp_str.yaml file"
    )

    parser.add_argument(
        "--fc_n_samples",
        type=int,
        default=None,
        help="Number of random displacement samples",
    )
    parser.add_argument(
        "--disp",
        type=float,
        default=0.001,
        help="Displacement (in Angstrom)",
    )
    parser.add_argument(
        "--is_plusminus",
        action="store_true",
        help="Plus-minus displacements will be generated.",
    )
    parser.add_argument(
        "--geometry_optimization",
        action="store_true",
        help="Geometry optimization is performed for initial structure.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=0.0,
        help="Pressure (in GPa)",
    )

    parser.add_argument(
        "--no_symmetry",
        action="store_true",
        help="Ignore symmetric properties in geometry optimization",
    )
    parser.add_argument(
        "--fix_cell",
        action="store_true",
        help="Fix cell shape and volume in geometry optimization",
    )
    parser.add_argument(
        "--fix_volume",
        action="store_true",
        help="Fix cell volume in geometry optimization",
    )
    parser.add_argument(
        "--fix_atom",
        action="store_true",
        help="Fix atomic positions in geometry optimization",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["BFGS", "CG", "L-BFGS-B", "SLSQP"],
        default="BFGS",
        help="Algorithm for geometry optimization",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size for FC solver.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Cutoff radius for setting zero elements.",
    )
    parser.add_argument(
        "--fc_orders",
        nargs="*",
        type=int,
        default=(2, 3),
        help="FC orders.",
    )

    parser.add_argument("--run_ltc", action="store_true", help="Run LTC calculations")
    parser.add_argument(
        "--ltc_mesh",
        type=int,
        nargs=3,
        default=[19, 19, 19],
        help="k-mesh used for phono3py calculation",
    )

    parser.add_argument(
        "--ph_mesh",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="k-mesh used for phonon calculation",
    )
    parser.add_argument("--ph_tmin", type=float, default=0, help="Temperature (min)")
    parser.add_argument("--ph_tmax", type=float, default=1000, help="Temperature (max)")
    parser.add_argument("--ph_tstep", type=float, default=10, help="Temperature (step)")
    parser.add_argument("--ph_pdos", action="store_true", help="Compute phonon PDOS")

    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        default=None,
        help="Input file name",
    )
    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()
    args = check_variables(args)

    if args.pot is None and args.infile is None:
        raise RuntimeError("Input parameters not found.")

    require_mlp = True if args.pot is not None else False
    polymlp = PypolymlpCalc(pot=args.pot, verbose=True, require_mlp=require_mlp)

    if args.properties:
        print("Mode: Property calculations", flush=True)
        polymlp.load_structures_from_files(
            poscars=args.poscars,
            vaspruns=args.vaspruns,
        )
        t1 = time.time()
        energies, forces, stresses = polymlp.eval()
        t2 = time.time()
        polymlp.save_properties()
        if len(forces) == 1:
            polymlp.print_properties()
        print("Elapsed time:", t2 - t1, "(s)", flush=True)

    elif args.force_constants:
        print("Mode: Force constant calculations", flush=True)
        supercell_matrix = np.diag(args.supercell)
        polymlp.load_poscars(args.poscar)
        if args.geometry_optimization:
            polymlp.init_geometry_optimization(
                with_sym=True,
                relax_cell=False,
                relax_volume=False,
                relax_positions=True,
            )
            polymlp.run_geometry_optimization()

        polymlp.init_fc(supercell_matrix=supercell_matrix, cutoff=args.cutoff)
        polymlp.run_fc(
            n_samples=args.fc_n_samples,
            distance=args.disp,
            is_plusminus=args.is_plusminus,
            orders=args.fc_orders,
            batch_size=args.batch_size,
            is_compact_fc=True,
            use_mkl=True,
        )
        polymlp.save_fc()

        if args.run_ltc:
            import phono3py

            ph3 = phono3py.load(
                unitcell_filename=args.poscar,
                supercell_matrix=supercell_matrix,
                primitive_matrix="auto",
                log_level=True,
            )
            ph3.mesh_numbers = args.ltc_mesh
            ph3.init_phph_interaction()
            ph3.run_thermal_conductivity(
                temperatures=range(0, 1001, 10),
                write_kappa=True,
            )

    elif args.geometry_optimization:
        print("Mode: Geometry optimization", flush=True)
        polymlp.load_poscars(args.poscar)
        relax_cell, relax_volume = True, True
        if args.fix_cell:
            relax_cell = False
            relax_volume = False
        if args.fix_volume:
            relax_volume = False
        polymlp.init_geometry_optimization(
            with_sym=not args.no_symmetry,
            relax_cell=relax_cell,
            relax_volume=relax_volume,
            relax_positions=not args.fix_atom,
            pressure=args.pressure,
        )
        polymlp.run_geometry_optimization(method=args.method)
        polymlp.save_poscars(filename="POSCAR_eqm")

    elif args.phonon:
        print("Mode: Phonon calculations", flush=True)
        supercell_matrix = np.diag(args.supercell)
        polymlp.load_poscars(args.poscar)

        polymlp.init_phonon(supercell_matrix=supercell_matrix)
        polymlp.run_phonon(
            distance=args.disp,
            mesh=args.ph_mesh,
            t_min=args.ph_tmin,
            t_max=args.ph_tmax,
            t_step=args.ph_tstep,
            with_eigenvectors=False,
            is_mesh_symmetry=True,
            with_pdos=args.ph_pdos,
        )
        polymlp.write_phonon()

        print("Mode: Phonon calculations (QHA)", flush=True)
        polymlp.run_qha(
            supercell_matrix=supercell_matrix,
            distance=args.disp,
            mesh=args.ph_mesh,
            t_min=args.ph_tmin,
            t_max=args.ph_tmax,
            t_step=args.ph_tstep,
            eps_min=0.8,
            eps_max=1.2,
            eps_step=0.02,
        )
        polymlp.write_qha()

    elif args.features or args.precision:
        print("Mode: Feature matrix calculations", flush=True)
        polymlp.load_structures_from_files(
            poscars=args.poscars,
            vaspruns=args.vaspruns,
        )
        polymlp.run_features(
            develop_infile=args.infile,
            features_force=False,
            features_stress=False,
        )

        if args.features:
            polymlp.save_features()
            print("features.npy is generated.", flush=True)

        if args.precision:
            print("Mode: Precision calculations", flush=True)
            prec = precision(polymlp.features)
            print(
                " precision, size (features):",
                prec,
                polymlp.features.shape,
                flush=True,
            )

    elif args.eos:
        if args.poscar is None and args.poscars is not None:
            args.poscar = args.poscars
        print("Mode: EOS calculation", flush=True)
        polymlp.load_poscars(args.poscar)
        polymlp.run_eos(
            eps_min=0.7,
            eps_max=2.0,
            eps_step=0.03,
            fine_grid=True,
            eos_fit=True,
        )
        polymlp.write_eos(filename="polymlp_eos.yaml")

    elif args.elastic:
        print("Mode: Elastic constant calculation", flush=True)
        polymlp.load_poscars(args.poscar)
        polymlp.run_elastic_constants()
        polymlp.write_elastic_constants(filename="polymlp_elastic.yaml")
