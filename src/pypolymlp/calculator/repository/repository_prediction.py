#!/usr/bin/env python 
import numpy as np
import glob, os
import yaml

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.repository.pypolymlp_calc import run_single_structure

from pypolymlp.calculator.repository.utils.target_structures import (
    get_structure_list_element1,
    get_structure_list_alloy2,
)
from pypolymlp.calculator.repository.utils.target_prototypes import (
    get_icsd_data1,
)
from pypolymlp.calculator.repository.utils.yaml_io import (
    write_icsd_yaml,
)

class PolymlpRepositoryPrediction:

    def __init__(self, 
                 yamlfile='polymlp_summary_convex.yaml',
                 path_mlp='./',
                 path_vasp='./'):

        yamldata = yaml.safe_load(open(yamlfile))['polymlps']

        self.__pot_dict = dict()
        for potdata in yamldata:
            path_pot = path_mlp + potdata['id'] 
            pot = sorted(glob.glob(path_pot + '/polymlp.lammps*'))
            self.__pot_dict[potdata['id']] = Properties(pot=pot)

        '''Finding elements automatically'''
        params_dict = list(self.__pot_dict.values())[0].params_dict
        self.__elements = params_dict['elements']

        if len(self.__elements) == 1:
            self.__target_list = get_structure_list_element1(
                                    self.__elements, path_vasp
                                 )
            self.__icsd_list = get_icsd_data1(self.__elements, path_vasp)
        else:
            raise ValueError('not available for more than binary system')

    def run_icsd(self, path_output='./'):

        print('--- Running ICSD prediction ---')
        for pot_id, prop in self.__pot_dict.items():
            print('Polymlp:', pot_id)
            st_dicts = [target['structure'] 
                        for _, target in self.__icsd_list.items()]
            energies, _, _ = prop.eval_multiple(st_dicts)
            energies = [e / sum(v['structure']['n_atoms']) 
                        for e, v in zip(energies, self.__icsd_list.values())]

            path_output_single = '/'.join([path_output, pot_id]) + '/'
            write_icsd_yaml(
                self.__icsd_list, energies, path_output=path_output_single
            )

            # plot_icsd

        return self


    def run(self, path_output='./'):

        self.run_icsd()

        print('--- Running property prediction ---')
        for pot_id, prop in self.__pot_dict.items():
            for st, target in self.__target_list.items():
                print('--- Polymlp:', pot_id, '(Structure:', st + ') ---')
                path_output_single = '/'.join([path_output, pot_id, st]) + '/'
                run_single_structure(target['structure'], 
                                     properties=prop,
                                     run_qha=False,
                                     path_output=path_output_single)

        return self


    def plot(self, path_output='./'):
        return self



  
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', 
                        type=str, 
                        default='polymlp_summary_convex.yaml',
                        help='Summary yaml file from grid search')
    parser.add_argument('--path_mlp', 
                        type=str, 
                        default='../4-grid/',
                        help='Path (regression data from grid search)')
    parser.add_argument('--path_vasp', 
                        type=str, 
                        default='./',
                        help='Path (vasp data for prototype structures)')
    parser.add_argument('--path_output', 
                        type=str, 
                        default='./',
                        help='Path (output of predictions)')
    args = parser.parse_args()

    pred = PolymlpRepositoryPrediction(yamlfile=args.yaml,
                                       path_mlp=args.path_mlp,
                                       path_vasp=args.path_vasp)
    pred.run()
    pred.plot()


