"""Functions used for running command line calculations."""

import time
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.utils import precision


def _set_structure(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
):
    """Set structure."""
    if structure is None:
        polymlp.load_poscars(args.poscar)
    else:
        polymlp.structures = structure
    return polymlp


def run_geometry_optimization(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
    filename: str = "POSCAR_eqm",
):
    """Run geometry optimization."""
    polymlp = _set_structure(args, polymlp, structure=structure)

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
    polymlp.run_geometry_optimization(
        method=args.method,
        gtol=args.gtol,
        maxiter=args.maxiter,
    )
    polymlp.save_poscars(filename=filename)
    return polymlp


def run_elastic_temperature(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
    filename: str = "polymlp_elastic_sscha.yaml",
):
    """Run temperature dependent elastic constant calculation."""
    polymlp = _set_structure(args, polymlp, structure=structure)
    polymlp.run_elastic_constants_temperature(gtol=args.gtol)
    polymlp.write_elastic_constants(filename=filename)
    return


def run_elastic(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
    filename: str = "polymlp_elastic.yaml",
):
    """Run elastic constant calculation."""
    polymlp = _set_structure(args, polymlp, structure=structure)
    polymlp.run_elastic_constants()
    polymlp.write_elastic_constants(filename=filename)


def run_eos(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
    filename: str = "polymlp_eos.yaml",
):
    """Run EOS calculation."""
    if args.poscar is None and args.poscars is not None:
        args.poscar = args.poscars

    polymlp = _set_structure(args, polymlp, structure=structure)
    polymlp.run_eos(
        eps_min=0.7,
        eps_max=2.0,
        eps_step=0.03,
        fine_grid=True,
        eos_fit=True,
    )
    polymlp.write_eos(filename=filename)


def run_phonon(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
):
    """Run phonon calculation."""
    polymlp = _set_structure(args, polymlp, structure=structure)
    supercell_matrix = np.diag(args.supercell)
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


def run_gsfe(
    args,
    polymlp: PypolymlpCalc,
    structure: Optional[PolymlpStructure] = None,
    filename: str = "gsfe.dat",
):
    """Run GSFE calculation."""
    polymlp = _set_structure(args, polymlp, structure=structure)
    excess_energies = polymlp.run_gsfe(
        disp1=args.disp1,
        disp2=args.disp2,
        glide_plane=args.glide,
        n_layers=args.n_layers,
        n_points=args.n_points,
        gtol=args.gtol,
        maxiter=args.maxiter,
        frac1=args.frac1,
        frac2=args.frac2,
        filename=filename,
    )
    return excess_energies


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
        run_geometry_optimization(args, polymlp)
    if args.elastic:
        print("Mode: Elastic constant calculation", flush=True)
        run_elastic(args, polymlp)
    if args.gsfe:
        print("Mode: GSFE calculation", flush=True)
        run_gsfe(args, polymlp)
    if args.eos:
        print("Mode: EOS calculation", flush=True)
        run_eos(args, polymlp)
    if args.phonon:
        print("Mode: Phonon calculations", flush=True)
        run_phonon(args, polymlp)

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

    if not calc_features:
        return

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
            prefix = " precision, size (features):"
            print(prefix, prec, polymlp.features.shape, flush=True)
