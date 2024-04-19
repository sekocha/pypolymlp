#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_gen.generator import (
        run_generator_single_dataset
)
from pypolymlp.mlp_gen.multi_datasets.generator import (
        run_generator_multiple_datasets
)
from pypolymlp.mlp_gen.multi_datasets.generator_sequential import (
        run_sequential_generator_multiple_datasets
)
from pypolymlp.mlp_gen.multi_datasets.additive.generator import (
        run_generator_additive
)
from pypolymlp.mlp_gen.multi_datasets.additive.generator_sequential import (
        run_sequential_generator_additive
)

from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.interface_vasp import parse_structures_from_vaspruns
from pypolymlp.calculator.compute_features import (
        compute_from_infile,
        compute_from_polymlp_lammps,
)
from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.yaml_utils import load_cells
from pypolymlp.core.utils import precision

def set_structures(args):

    if args.poscars is not None:
        print('Loading POSCAR files:')
        for p in args.poscars:
            print('-', p)
        structures = parse_structures_from_poscars(args.poscars)
    elif args.vaspruns is not None:
        print('Loading vasprun.xml files:')
        for v in args.vaspruns:
            print('-', v)
        structures = parse_structures_from_vaspruns(args.vaspruns)
    elif args.phono3py_yaml is not None:
        from pypolymlp.core.interface_phono3py import (
            parse_structures_from_phono3py_yaml
        )
        print('Loading', args.phono3py_yaml)
        if args.phono3py_yaml_structure_ids is not None:
            r1, r2 = args.phono3py_yaml_structure_ids
            select_ids = np.arange(r1, r2)
        else:
            select_ids = None

        structures = parse_structures_from_phono3py_yaml(
                args.phono3py_yaml,
                select_ids=select_ids)

    return structures

def compute_features(structures, args, force=False):
    if args.pot is not None:
        return compute_from_polymlp_lammps(structures, 
                                           pot=args.pot,
                                           return_mlp_dict=False,
                                           force=force)
    infile = args.infile[0]
    return compute_from_infile(infile, structures, force=force)

def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        nargs='*',
                        type=str, 
                        default=None,
                        help='Input file name')
    parser.add_argument('--sequential', 
                        action='store_true',
                        help='Use sequential evaluation of X.T @ X')
    parser.add_argument('--learning_curve', 
                        action='store_true',
                        help='Compute learning curve')
    parser.add_argument('--features', 
                        action='store_true',
                        help='Mode: Feature calculation')
    parser.add_argument('--properties', 
                        action='store_true',
                        help='Mode: Property calculation')
    parser.add_argument('--force_constants', 
                        action='store_true',
                        help='Mode: Force constant calculation')
    parser.add_argument('--phonon', 
                        action='store_true',
                        help='Mode: Phonon calculation')
    parser.add_argument('--precision', 
                        action='store_true',
                        help=('Mode: MLP precision calculation.',
                              'This uses only features'))

    parser.add_argument('--pot',
                        nargs='*',
                        type=str,
                        default=None,
                        help='polymlp file')
    parser.add_argument('--poscars',
                        nargs='*',
                        type=str,
                        default=None,
                        help='poscar files')
    parser.add_argument('--vaspruns',
                        nargs='*',
                        type=str,
                        default=None,
                        help='vasprun files')
    parser.add_argument('--phono3py_yaml',
                        type=str,
                        default=None,
                        help='phono3py.yaml file')
    parser.add_argument('--phono3py_yaml_structure_ids',
                        nargs=2,
                        type=int,
                        default=None,
                        help='Structure range in phono3py.yaml file')

    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')
    parser.add_argument('--str_yaml',
                        type=str,
                        default=None,
                        help='polymlp_str.yaml file')

    parser.add_argument('--fc_n_samples',
                        type=int,
                        default=None,
                        help='Number of random displacement samples')
    parser.add_argument('--disp',
                        type=float,
                        default=0.03,
                        help='Displacement (in Angstrom)')
    parser.add_argument('--is_plusminus',
                        action='store_true',
                        help='Plus-minus displacements will be generated.')
    parser.add_argument('--geometry_optimization',
                        action='store_true',
                        help='Geometry optimization is performed '
                             'for initial structure.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=200,
                        help='Batch size for FC solver.')

    parser.add_argument('--ph_mesh',
                        type=int,
                        nargs=3,
                        default=[10,10,10],
                        help='k-mesh used for phonon calculation')
    parser.add_argument('--ph_tmin',
                        type=float,
                        default=100,
                        help='Temperature (min)')
    parser.add_argument('--ph_tmax',
                        type=float,
                        default=1000,
                        help='Temperature (max)')
    parser.add_argument('--ph_tstep',
                        type=float,
                        default=100,
                        help='Temperature (step)')
    parser.add_argument('--ph_pdos',
                        action='store_true',
                        help='Compute phonon PDOS')
    args = parser.parse_args()


    mode_regression = False
    if args.infile is not None: 
        if args.features == False and args.precision == False:
            mode_regression = True

    if mode_regression:
        if len(args.infile) == 1:
            infile = args.infile[0]
            params_dict = ParamsParser(infile).get_params()
            if params_dict['dataset_type'] == 'phono3py':
                print('Mode: Regression')
                run_generator_single_dataset(infile, 
                    compute_learning_curve=args.learning_curve)
            else:
                if args.sequential:
                    print('Mode: Sequential regression')
                    run_sequential_generator_multiple_datasets(infile)
                elif args.learning_curve: 
                    print('Mode: Regression and learning curve')
                    run_generator_single_dataset(infile, 
                                                 compute_learning_curve=True)
                else:
                    print('Mode: Regression')
                    run_generator_multiple_datasets(infile)
        else:
            if args.sequential:
                print('Mode: Sequential regression (hybrid model)')
                run_sequential_generator_additive(args.infile)
            else:
                print('Mode: Regression (hybrid model)')
                run_generator_additive(args.infile)

    if args.properties:
        print('Mode: Property calculations')
        structures = set_structures(args)
        prop = Properties(pot=args.pot)
        energies, forces, stresses = prop.eval_multiple(structures)
        stresses_gpa = convert_stresses_in_gpa(stresses, structures)

        np.set_printoptions(suppress=True)
        np.save('polymlp_energies.npy', energies)
        np.save('polymlp_forces.npy', forces)
        np.save('polymlp_stress_tensors.npy', stresses_gpa)

        if len(forces) == 1:
            np.savetxt('polymlp_energies.dat', energies, fmt='%f')
            print(' energy =', energies[0], '(eV/cell)')
            print(' forces =')
            for i, f in enumerate(forces[0].T):
                print('  - atom', i, ":", f)
            stress = stresses_gpa[0]
            print(' stress tensors =')
            print('  - xx, yy, zz:', stress[0:3])
            print('  - xy, yz, zx:', stress[3:6])
            print('---------')

        print('polymlp_energies.npy, polymlp_forces.npy,',
                'and polymlp_stress_tensors.npy are generated.')

    elif args.force_constants:
        from pypolymlp.calculator.compute_fcs import (
            compute_fcs_from_structure,
            compute_fcs_phono3py_dataset,
        )
        print('Mode: Force constant calculations')
        if args.phono3py_yaml is not None:
            compute_fcs_phono3py_dataset(
                pot=args.pot,
                phono3py_yaml=args.phono3py_yaml,
                use_phonon_dataset=False,
                n_samples=args.fc_n_samples,
                displacements=args.disp,
                is_plusminus=args.is_plusminus,
                geometry_optimization=args.geometry_optimization,
                batch_size=args.batch_size,
            )


        else:
            if args.str_yaml is not None:
                _, supercell_dict = load_cells(filename=args.str_yaml)
                unitcell_dict = None
                supercell_matrix = None
            elif args.poscar is not None:
                unitcell_dict = Poscar(args.poscar).get_structure()
                supercell_matrix = np.diag(args.supercell)
                supercell_dict = None
   
            compute_fcs_from_structure(
                pot=args.pot,
                unitcell_dict=unitcell_dict,
                supercell_dict=supercell_dict,
                supercell_matrix=supercell_matrix,
                n_samples=args.fc_n_samples,
                displacements=args.disp,
                is_plusminus=args.is_plusminus,
                geometry_optimization=args.geometry_optimization,
                batch_size=args.batch_size,
            )

    elif args.phonon:
        from pypolymlp.calculator.compute_phonon import (
            PolymlpPhonon, PolymlpPhononQHA
        )
        print('Mode: Phonon calculations')

        if args.str_yaml is not None:
            unitcell_dict, supercell_dict = load_cells(filename=args.str_yaml)
            supercell_matrix = supercell_dict['supercell_matrix']
        elif args.poscar is not None:
            unitcell_dict = Poscar(args.poscar).get_structure()
            supercell_matrix = np.diag(args.supercell)

        ph = PolymlpPhonon(unitcell_dict, supercell_matrix, pot=args.pot)
        ph.produce_force_constants(displacements=args.disp)
        ph.compute_properties(mesh=args.ph_mesh,
                              t_min=args.ph_tmin,
                              t_max=args.ph_tmax,
                              t_step=args.ph_tstep,
                              pdos=args.ph_pdos)


        qha = PolymlpPhononQHA(unitcell_dict, supercell_matrix, pot=args.pot)
        qha.run()

    elif args.features:
        print('Mode: Feature matrix calculations')
        structures = set_structures(args)
        x = compute_features(structures, args, force=False)
        print(' feature size =', x.shape)
        np.save('features.npy', x)
        print('features.npy is generated.')

    elif args.precision:
        print('Mode: Precision calculations')
        structures = set_structures(args)
        x = compute_features(structures, args, force=True)
        prec = precision(x)
        print(' precision, size (features):', prec, x.shape)


