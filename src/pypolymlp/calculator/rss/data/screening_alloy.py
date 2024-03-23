#!/usr/bin/env python 
import numpy as np
import os
import argparse
import signal
import warnings

from pypolymlp.calculator.rss.data.common import (
    parse_log_summary_yaml, estimate_n_locals, set_emin,
)
from pypolymlp.calculator.rss.data.alloy import AlloyEnergy


def write_hull_yaml(data_ch, e_form_ch, e_info, filename='log_hull.yaml'):

    data = sorted([[tuple(d[:-1]), d[-1], ef]
                  for d, ef in zip(data_ch, e_form_ch)], reverse=True)

    f = open(filename, 'w')
    print('nonequiv_structures:', file=f)
    for comp, e, ef in data:
        print('- composition:', list(comp), file=f)
        print('  structures:', file=f)
        print('  - id:         ', e_info[comp]['id_min'], file=f)
        print('    e:          ', "{:.5f}".format(e), file=f)
        print('    e_formation:', "{:.5f}".format(ef), file=f)
        print('', file=f)
    f.close()



if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file',
                        type=str,
                        default='log_summary.yaml',
                        help='summary yaml file')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.yaml_file)
    summary_dict, numbers = parse_log_summary_yaml(args.yaml_file, 
                                                   return_numbers=True)
    n_trials = numbers['number_of_trial_structures']

    e_info = dict()
    data_min = []
    for comp, summary in summary_dict.items():
        e_all = summary['energies']
        e_min, e_outlier = set_emin(e_all)
        match = e_all > e_outlier
        id_min = np.argmin(e_all[match])

        e_info[comp] = dict()
        e_info[comp]['min'] = e_min
        e_info[comp]['outlier'] = e_outlier
        e_info[comp]['id_min'] = summary['poscars'][id_min]

        if e_outlier > -np.inf:
            print('Composition =', comp)
            print(' Energy min. =', e_min)
            print(' Energy threshold (outlier) =', e_outlier)
            print(' Energy (< e_outlier) =', e_all[e_all < e_outlier])

        add_entry = list(comp)
        add_entry.append(e_min)
        data_min.append(add_entry)
    data_min = np.array(data_min).astype(float)

    alloy = AlloyEnergy()
    data_ch, ids_ch = alloy.compute_ch(data_min=data_min)
    e_form_ch = alloy.compute_formation_e(data_ch)
    write_hull_yaml(data_ch, e_form_ch, e_info)

    '''Evaluating E_above_convex_hull'''
    alloy.initialize_composition_partition()
    for comp_tuple, summary in summary_dict.items():
        e_all = summary['energies']
        id_all = summary['poscars']
        comp = np.array(comp_tuple).astype(float)

        print('C', comp)
        e_form = alloy.compute_formation_e2(comp, e_all)
        print(e_all[:5])
        e_hull = alloy.get_convex_hull_energy(comp)
        print(e_hull)



    



#    for comp, summary in summary_dict.items():
#        e_list = [0.005,0.01,0.015,0.02] + list(np.arange(0.025,0.301,0.025))
#        for e_th in e_list:
#            e_ub = e_min + e_th
#            indices = np.where(e_all < e_ub)[0]
#            spg_array = [summary['space_groups'][i] for i in indices]
#
#            n_locals_estimator = estimate_n_locals(n_trials, len(indices))
#
#            f = open(output_dir + '/log_screening_' 
#                     + str(round(e_th,6)).ljust(5,'0') + '.yaml','w')
#            print('numbers:', file=f)
#            print('  number_of_local_minimizers:', len(indices), file=f)
#            print('  number_of_locals_estimator:', 
#                    n_locals_estimator, file=f)
#            print('  threshold_energy_(eV/atom):', round(e_th,6), file=f)
#            print('', file=f)
#
#            print('nonequiv_structures:', file=f)
#            print('- composition:', list(comp), file=f)
#            print('  structures:', file=f)
#            for poscar, e, spg in zip(summary['poscars'][indices], 
#                                      summary['energies'][indices],
#                                      spg_array):
#                print('  - id:  ', poscar, file=f)
#                print('    e:   ', e, file=f)
#                print('    spg: ', spg, file=f)
#            f.close()
#                
#
