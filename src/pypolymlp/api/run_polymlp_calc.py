"""Command lines for calculating properites using polynomial MLP."""

import argparse
import signal
import time

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.utils import precision, print_credit

from .common_args import (
    create_fc_parser,
    create_go_parser,
    create_mode_parser,
    create_phonon_parser,
    create_polymlp_parser,
    create_structure_parser,
)


def check_variables(args):
    """Check variables."""
    if args.poscar is None and args.poscars is not None:
        args.poscar = args.poscars
    if args.poscars is None and args.poscar is not None:
        args.poscars = args.poscar
    return args


def run_calculations(args, polymlp: PypolymlpCalc, calc_features: bool = True):
    """Run calculations."""
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
            try:
                polymlp.print_properties()
            except:
                pass
        print("Elapsed time:", t2 - t1, "(s)", flush=True)

    if args.geometry_optimization:
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

    if args.eos:
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

    if args.elastic:
        print("Mode: Elastic constant calculation", flush=True)
        polymlp.load_poscars(args.poscar)
        polymlp.run_elastic_constants()
        polymlp.write_elastic_constants(filename="polymlp_elastic.yaml")

    if args.force_constants:
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

        cutoff = {2: args.cutoff_fc2, 3: args.cutoff_fc3, 4: args.cutoff_fc4}
        polymlp.init_fc(supercell_matrix=supercell_matrix, cutoff=cutoff)
        polymlp.run_fc(
            n_samples=args.fc_n_samples,
            distance=args.disp,
            is_plusminus=args.is_plusminus,
            orders=args.fc_orders,
            batch_size=args.batch_size,
            is_compact_fc=True,
            use_mkl=True,
            use_gradient_solver=args.use_gradient_solver,
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

    if args.phonon:
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

    if calc_features:
        if args.features or args.precision:
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


def run():
    """Main code for command line."""

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    mode_parser, mode_group = create_mode_parser()
    mode_group.add_argument(
        "--features", action="store_true", help="Mode: Feature calculation"
    )

    polymlp_parser = create_polymlp_parser()
    st_parser = create_structure_parser(multiple=True, enable_yaml=True)
    fc_parser = create_fc_parser()
    go_parser = create_go_parser()
    phonon_parser = create_phonon_parser()

    parser = argparse.ArgumentParser(
        description="Calculations using PolyMLP",
        parents=[
            mode_parser,
            polymlp_parser,
            st_parser,
            go_parser,
            phonon_parser,
            fc_parser,
        ],
    )

    feature_group = parser.add_argument_group(
        "Features", "Options for structural feature calculation"
    )
    feature_group.add_argument(
        "-i",
        "--infile",
        type=str,
        default=None,
        help="Input file name",
    )
    feature_group.add_argument(
        "--precision",
        action="store_true",
        help="Mode: MLP precision calculation. This uses only features",
    )

    args = parser.parse_args()
    np.set_printoptions(legacy="1.21")
    print_credit()
    args = check_variables(args)

    if args.pot is None and args.infile is None:
        raise RuntimeError("Input parameters not found.")

    require_mlp = True if args.pot is not None else False
    polymlp = PypolymlpCalc(pot=args.pot, verbose=True, require_mlp=require_mlp)

    run_calculations(args, polymlp)
