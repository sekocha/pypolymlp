#!/usr/bin/env python
import argparse
import signal
import time

import numpy as np

from pypolymlp.calculator.compute_features import (
    compute_from_infile,
    compute_from_polymlp_lammps,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import (
    Poscar,
    parse_structures_from_poscars,
    parse_structures_from_vaspruns,
)
from pypolymlp.core.utils import precision
from pypolymlp.utils.yaml_utils import load_cells


def set_structures(args):

    if args.poscars is not None:
        print("Loading POSCAR files:")
        for p in args.poscars:
            print("-", p)
        structures = parse_structures_from_poscars(args.poscars)
    elif args.vaspruns is not None:
        print("Loading vasprun.xml files:")
        for v in args.vaspruns:
            print("-", v)
        structures = parse_structures_from_vaspruns(args.vaspruns)
    elif args.phono3py_yaml is not None:
        from pypolymlp.core.interface_phono3py import (
            parse_structures_from_phono3py_yaml,
        )

        print("Loading", args.phono3py_yaml)
        if args.phono3py_yaml_structure_ids is not None:
            r1, r2 = args.phono3py_yaml_structure_ids
            select_ids = np.arange(r1, r2)
        else:
            select_ids = None

        structures = parse_structures_from_phono3py_yaml(
            args.phono3py_yaml, select_ids=select_ids
        )

    return structures


def compute_features(structures, args, force=False):
    if args.pot is not None:
        return compute_from_polymlp_lammps(
            structures, pot=args.pot, return_mlp_dict=False, force=force
        )
    infile = args.infile[0]
    return compute_from_infile(infile, structures, force=force)


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
        default=0.03,
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
        help="Geometry optimization is performed " "for initial structure.",
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
    parser.add_argument("--ph_tmin", type=float, default=100, help="Temperature (min)")
    parser.add_argument("--ph_tmax", type=float, default=1000, help="Temperature (max)")
    parser.add_argument(
        "--ph_tstep", type=float, default=100, help="Temperature (step)"
    )
    parser.add_argument("--ph_pdos", action="store_true", help="Compute phonon PDOS")
    args = parser.parse_args()

    if args.properties:
        print("Mode: Property calculations")
        structures = set_structures(args)
        prop = Properties(pot=args.pot)
        t1 = time.time()
        energies, forces, stresses = prop.eval_multiple(structures)
        t2 = time.time()
        prop.save()
        if len(forces) == 1:
            prop.print_single()
        print("Elapsed time:", t2 - t1, "(s)")

    elif args.force_constants:
        from pypolymlp.calculator.fc import PolymlpFC
        from pypolymlp.utils.phonopy_utils import phonopy_supercell

        print("Mode: Force constant calculations")
        supercell = None
        if args.str_yaml is not None:
            _, supercell_dict = load_cells(filename=args.str_yaml)
            supercell_matrix = supercell_dict["supercell_matrix"]
            supercell = phonopy_supercell(supercell_dict, np.eye(3))
        elif args.poscar is not None:
            unitcell_dict = Poscar(args.poscar).get_structure()
            supercell_matrix = np.diag(args.supercell)
            supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

        polyfc = PolymlpFC(
            supercell=supercell,
            phono3py_yaml=args.phono3py_yaml,
            use_phonon_dataset=False,
            pot=args.pot,
            cutoff=args.cutoff,
        )
        if args.geometry_optimization:
            polyfc.run_geometry_optimization()

        if args.fc_n_samples is not None:
            polyfc.sample(
                n_samples=args.fc_n_samples,
                displacements=args.disp,
                is_plusminus=args.is_plusminus,
            )

        polyfc.run(write_fc=True, batch_size=args.batch_size)

        if args.run_ltc:
            import phono3py

            ph3 = phono3py.load(
                unitcell_filename=args.poscar,
                supercell_matrix=supercell_matrix,
                primitive_matrix="auto",
                log_level=1,
            )
            ph3.mesh_numbers = args.ltc_mesh
            ph3.init_phph_interaction()
            ph3.run_thermal_conductivity(
                temperatures=range(0, 1001, 10),
                write_kappa=True,
            )

    elif args.phonon:
        from pypolymlp.calculator.compute_phonon import PolymlpPhonon, PolymlpPhononQHA

        print("Mode: Phonon calculations")

        if args.str_yaml is not None:
            unitcell_dict, supercell_dict = load_cells(filename=args.str_yaml)
            supercell_matrix = supercell_dict["supercell_matrix"]
        elif args.poscar is not None:
            unitcell_dict = Poscar(args.poscar).get_structure()
            supercell_matrix = np.diag(args.supercell)

        ph = PolymlpPhonon(unitcell_dict, supercell_matrix, pot=args.pot)
        ph.produce_force_constants(displacements=args.disp)
        ph.compute_properties(
            mesh=args.ph_mesh,
            pdos=args.ph_pdos,
            t_min=args.ph_tmin,
            t_max=args.ph_tmax,
            t_step=args.ph_tstep,
        )

        qha = PolymlpPhononQHA(unitcell_dict, supercell_matrix, pot=args.pot)
        qha.run()
        qha.write_qha()

    elif args.features:
        print("Mode: Feature matrix calculations")
        structures = set_structures(args)
        x = compute_features(structures, args, force=False)
        print(" feature size =", x.shape)
        np.save("features.npy", x)
        print("features.npy is generated.")

    elif args.precision:
        print("Mode: Precision calculations")
        structures = set_structures(args)
        x = compute_features(structures, args, force=True)
        prec = precision(x)
        print(" precision, size (features):", prec, x.shape)


# if __name__ == '__main__':
#    run()
