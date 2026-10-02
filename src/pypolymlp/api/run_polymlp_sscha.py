"""Command lines for performing SSCHA calculations by command line."""

import argparse
import signal

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.utils import print_credit

from .common_args import (
    create_advanced_sscha_parser,
    create_go_parser,
    create_polymlp_parser,
    create_sscha_parser,
    create_structure_parser,
)
from .run_polymlp_calc import run_geometry_optimization


def run_main_sscha(args, polymlp: PypolymlpCalcProperties):
    """Run SSCHA calculations."""

    #     if args.yaml is not None:
    #         sscha.load_restart(yaml=args.yaml, parse_fc2=True)
    #     elif args.poscar is not None:
    #         sscha.load_poscar(args.poscar, np.diag(args.supercell))
    #     else:
    #         raise RuntimeError("Structure not found. Use --poscar or --yaml option.")
    #
    #     if args.born_vasprun is not None:
    #         sscha.set_nac_params(args.born_vasprun)
    #
    if args.n_samples is None:
        n_samples_init, n_samples_final = None, None
    else:
        n_samples_init, n_samples_final = args.n_samples

    unitcell = polymlp.load_poscars(args.poscar)
    supercell_matrix = np.diag(args.supercell)

    prop = polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=supercell_matrix,
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

    calc = PypolymlpCalc(properties=prop, verbose=True)
    if args.geometry_optimization:
        if args.temp is None:
            raise RuntimeError("Temperature required. Use --temp option.")

        print("Mode: SSCHA geometry optimization", flush=True)
        run_geometry_optimization(args, calc)

    elif args.elastic:
        if args.temp is None:
            raise RuntimeError("Temperature required. Use --temp option.")

        print("Mode: SSCHA elastic constant calculation", flush=True)
        calc.load_poscars(args.poscar)
        calc.run_elastic_constants_temperature(gtol=args.gtol)
        calc.write_elastic_constants(filename="polymlp_elastic_sscha.yaml")
    else:
        print("Mode: SSCHA calculation", flush=True)
        calc = PypolymlpCalc(properties=prop, verbose=True)
        free_energy, _, _ = calc.eval(unitcell)


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

    run_main_sscha(args, polymlp)
