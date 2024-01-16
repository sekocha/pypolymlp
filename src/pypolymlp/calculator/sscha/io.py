#!/usr/bin/env python
import numpy as np
import os
import yaml

def temperature_setting(args):

    if args.temp is not None:
        temp_array = [args.temp]
    else:
        temp_array = np.arange(args.temp_min, args.temp_max+1, args.temp_step)
        if args.ascending_temp == False:
            temp_array = temp_array[::-1]
    args.temperatures = temp_array
    return args


def n_steps_setting(args, n_atom_supercell=None):

    if args.n_samples is None:
        n_steps_unit = round(3200/n_atom_supercell)
        args.n_steps = 20 * n_steps_unit
        args.n_steps_final = 100 * n_steps_unit
    else:
        args.n_steps, args.n_steps_final = args.n_samples
    return args


def print_structure(cell):

    print(' # structure ')
    print('  - elements:     ', cell['elements'])
    print('  - axis:         ', cell['axis'].T[0])
    print('                  ', cell['axis'].T[1])
    print('                  ', cell['axis'].T[2])
    print('  - positions:    ', cell['positions'].T[0])
    if cell['positions'].shape[1] > 1:
        for pos in cell['positions'].T[1:]:
            print('                  ', pos)


def print_parameters(supercell, args):

    print(' # parameters')
    print('  - supercell:    ', supercell[0])
    print('                  ', supercell[1])
    print('                  ', supercell[2])
    print('  - temperatures: ', args.temperatures[0])
    if len(args.temperatures) > 1:
        for t in args.temperatures[1:]:
            print('                  ', t)

    print('  - Polynomial ML potential:  ', os.path.abspath(args.pot))
    print('  - FC tolerance:             ', args.tol)
    print('  - max iter:                 ', args.max_iter)
    print('  - num samples:              ', args.n_steps)
    print('  - num samples (last iter.): ', args.n_steps_final)
    print('  - q-mesh:                   ', args.mesh)


def print_array1d(array, tag, fstream, indent_l=0):
    prefix = ''.join([' ' for n in range(indent_l)])
    print(prefix + tag + ':', file=fstream)
    for i, d in enumerate(array):
        print(prefix + ' -', d, file=fstream)


def print_array2d(array, tag, fstream, indent_l=0):
    prefix = ''.join([' ' for n in range(indent_l)])
    print(prefix + tag + ':', file=fstream)
    for i, d in enumerate(array):
        print(prefix + ' -', list(d), file=fstream)


def save_cell(cell, tag='unitcell', fstream=None, filename=None):

    if fstream is None:
        fstream = open(filename, 'w')

    print(tag+':', file=fstream)
    print_array2d(cell['axis'].T, 'axis', fstream, indent_l=2)
    print_array2d(cell['positions'].T, 'positions', fstream, indent_l=2)
    print('  n_atoms:  ', list(cell['n_atoms']), file=fstream)
    print('  types:    ', list(cell['types']), file=fstream)
    print('  elements: ', list(cell['elements']), file=fstream)

    if tag == 'supercell':
        print('  n_unitcells: ', cell['n_unitcells'], file=fstream)
        print_array2d(cell['supercell_matrix'],
                      'supercell_matrix', fstream, indent_l=2)

    print('', file=fstream)


def save_sscha_results(sscha_dict, 
                       log_dict, 
                       unitcell, 
                       supercell_matrix,
                       args,
                       filename='sscha_results.yaml'):

    f = open(filename, 'w')
    print('parameters:', file=f)
    print('  pot:     ', os.path.abspath(args.pot), file=f)
    print('  temperature:   ', sscha_dict['temperature'], file=f)
    print('  n_steps:       ', args.n_steps, file=f)
    print('  n_steps_final: ', args.n_steps_final, file=f)
    print('  tolerance:     ', args.tol, file=f)
    print('  mixing:        ', args.mixing, file=f)
    print('  mesh_phonon:   ', list(args.mesh), file=f)
    print('', file=f)

    print('properties:', file=f)
    print('  free_energy:', sscha_dict['free_energy'], file=f)

    print('status:', file=f)
    print('  delta_fc: ', log_dict['delta'], file=f)
    print('  converge: ', log_dict['converge'], file=f)
    print('', file=f)

    save_cell(unitcell, tag='unitcell', fstream=f)
    print('supercell_matrix:', file=f)
    print(' -', list(supercell_matrix[0]), file=f)
    print(' -', list(supercell_matrix[1]), file=f)
    print(' -', list(supercell_matrix[2]), file=f)
 
#    print('logs:', file=f)
#    logs = sscha_dict['logs']
#    print_array1d(logs['f_sscha'], 'free_energy', f, indent_l=2)
#    print('', file=f)
#    print_array1d(logs['f_harmonic'], 'harmonic_free_energy', f, indent_l=2)
#    print('', file=f)
#    print_array1d(logs['pot_harmonic'], 'harmonic_potential', f, indent_l=2)
#    print('', file=f)
#    print_array1d(logs['pot_ensemble'], 'ensemble_potential', f, indent_l=2)
#    print('', file=f)

    f.close()

def load_sscha_results(res_yaml):

    yaml_data = yaml.safe_load(open(res_yaml))

    parameters = yaml_data['parameters']
    sscha_dict = yaml_data['properties']
    status_dict = yaml_data['status']
#    logs = yaml_data['logs']
    logs = None

    unitcell = yaml_data['unitcell']
    unitcell['axis'] = np.array(unitcell['axis']).T
    unitcell['positions'] = np.array(unitcell['positions']).T

    supercell_matrix = np.array(yaml_data['supercell_matrix'])

    return ((parameters, sscha_dict, status_dict, logs),
            (unitcell, supercell_matrix))


