"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal

import numpy as np

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA
from pypolymlp.core.utils import print_credit

from .common_args import (
    create_advanced_sscha_parser,
    create_go_parser,
    create_polymlp_parser,
    create_sscha_parser,
    create_structure_parser,
)


def run_main_sscha(args, sscha: PypolymlpSSCHA):
    """Run SSCHA calculations."""
    if args.yaml is not None:
        sscha.load_restart(yaml=args.yaml, parse_fc2=True)
    elif args.poscar is not None:
        sscha.load_poscar(args.poscar, np.diag(args.supercell))
    else:
        raise RuntimeError("Structure not found. Use --poscar or --yaml option.")

    if args.born_vasprun is not None:
        sscha.set_nac_params(args.born_vasprun)

    if args.n_samples is None:
        n_samples_init, n_samples_final = None, None
    else:
        n_samples_init, n_samples_final = args.n_samples

    if args.geometry_optimization:
        print("Mode: SSCHA geometry optimization", flush=True)
        if args.temp is None:
            raise RuntimeError("Temperature required. Use --temp option.")

        relax_cell, relax_volume = True, True
        if args.fix_cell:
            relax_cell = False
            relax_volume = False
        if args.fix_volume:
            relax_volume = False

        sscha.run_geometry_optimization(
            temp=args.temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=args.tol,
            max_iter=args.max_iter,
            mixing=args.mixing,
            mesh=args.mesh,
            init_fc_algorithm=args.init,
            init_fc_file=args.init_file,
            cutoff_radius=args.cutoff_fc2,
            use_mkl=not args.disable_mkl,
            with_sym=not args.no_symmetry,
            relax_cell=relax_cell,
            relax_volume=relax_volume,
            relax_positions=not args.fix_atom,
            pressure=args.pressure,
            gtol=args.gtol,
        )
    elif args.elastic:
        if args.temp is None:
            raise RuntimeError("Temperature required. Use --temp option.")
        print("Mode: SSCHA elastic constant calculation", flush=True)
        sscha.run_elastic(
            temp=args.temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=args.tol,
            max_iter=args.max_iter,
            mixing=args.mixing,
            mesh=args.mesh,
            init_fc_algorithm=args.init,
            init_fc_file=args.init_file,
            cutoff_radius=args.cutoff_fc2,
            use_mkl=not args.disable_mkl,
            gtol=args.gtol,
            verbose_sscha=False,
        )
    else:
        print("Mode: SSCHA calculation", flush=True)
        sscha.run(
            temp=args.temp,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            temp_step=args.temp_step,
            n_temp=args.n_temp,
            ascending_temp=args.ascending_temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=args.tol,
            max_iter=args.max_iter,
            mixing=args.mixing,
            mesh=args.mesh,
            init_fc_algorithm=args.init,
            init_fc_file=args.init_file,
            cutoff_radius=args.cutoff_fc2,
            use_temporal_cutoff=args.use_temporal_cutoff,
            precondition=not args.disable_precondition,
            write_pdos=args.write_pdos,
            use_mkl=not args.disable_mkl,
        )


def run():

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

    sscha = PypolymlpSSCHA(verbose=True)
    if args.pot is not None:
        sscha.set_polymlp(args.pot)

    run_main_sscha(args, sscha)
