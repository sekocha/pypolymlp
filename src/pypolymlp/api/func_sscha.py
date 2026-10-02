"""Functions used for running command line calculations."""

import numpy as np

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

from .func_calc import run_geometry_optimization


def run_main_sscha(args, polymlp: PypolymlpCalcProperties):
    """Run SSCHA calculations."""
    if args.poscar is None and args.yaml is None:
        raise RuntimeError("Structure not found. Use --poscar or --yaml option.")
    if polymlp.static_calculator is None:
        raise RuntimeError("Static Properties Calculator not found.")

    #     if args.yaml is not None:
    #         sscha.load_restart(yaml=args.yaml, parse_fc2=True)
    #     elif args.poscar is not None:
    #         sscha.load_poscar(args.poscar, np.diag(args.supercell))
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
        free_energy, _, _ = calc.eval(unitcell)
    return
