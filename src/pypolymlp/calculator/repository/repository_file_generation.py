#!/usr/bin/env python 
import numpy as np
import os
import yaml
from collections import defaultdict

from pypolymlp.calculator.repository.utils.figure_utils_summary import (
    plot_mlp_distribution,
    plot_eqm_properties,
)

#from pypolymlp.calculator.repository.utils.figure_utils_each_pot import (
#    plot_mlp_distribution,
#)

#from pypolymlp.calculator.repository.utils.yaml_io import (
#    write_icsd_yaml,
#)

class PolymlpRepositoryGeneration:

    def __init__(self, path_data='./'):

        self.__path_data = path_data
        yaml_name = path_data + '/polymlp_summary_convex.yaml'
        yamldata_convex = yaml.safe_load(open(yaml_name))
        self.__system = yamldata_convex['system']
        self.__yamldata_convex = yamldata_convex['polymlps']

        self.__pot_ids = [d['id'] for d in self.__yamldata_convex]
        self.__costs = [float(d['cost_single']) for d in self.__yamldata_convex]

        yaml_name = path_data + '/polymlp_summary/prediction.yaml'
        yaml_data = yaml.safe_load(open(yaml_name))
        self.__target_list = [d['st_type'] for d in yaml_data['structures']]

    def run_mlp_distribution(self, dpi=300):

        yaml_name = self.__path_data + '/polymlp_summary_all.yaml'
        yamldata = yaml.safe_load(open(yaml_name))['polymlps']

        d_all = [[d['cost_single'], 
                  d['cost_openmp'], 
                  d['rmse_energy'], 
                  d['rmse_force'],
                  d['id']] for d in yamldata]
        d_convex = [[d['cost_single'], 
                     d['cost_openmp'], 
                     d['rmse_energy'], 
                     d['rmse_force'],
                     d['id']] for d in self.__yamldata_convex]
        d_all = np.array(d_all)
        d_convex = np.array(d_convex)

        path_output = self.__path_data + '/polymlp_summary/'
        plot_mlp_distribution(
            d_all, d_convex, self.__system, path_output=path_output, dpi=dpi,
        )

        return self
        
    def run_eos(self, dpi=300):

        eos_dict = defaultdict(dict)
        eqm_props_dict = dict()
        for st in self.__target_list:
            eqm_props = []
            for pot_id, cost in zip(self.__pot_ids, self.__costs):
                yamlfile = '/'.join(
                    [self.__path_data, pot_id, 'predictions', 
                     st, 'polymlp_eos.yaml']
                )
                yamldata = yaml.safe_load(open(yamlfile))

                eqm_data = yamldata['equilibrium']
                n_atom_sum = sum([int(n) for n in eqm_data['n_atoms']])
                energy = float(eqm_data['free_energy']) / n_atom_sum
                volume = float(eqm_data['volume']) / n_atom_sum
                bm = float(eqm_data['bulk_modulus'])
                eqm_props.append([cost, energy, volume, bm])

                data = yamldata['eos_data']['volume_helmholtz']
                data = np.array(data, dtype=float) / n_atom_sum
                eos_dict[pot_id][st] = data

            eqm_props_dict[st] = np.array(eqm_props)

        path_output = self.__path_data + '/polymlp_summary/'
        plot_eqm_properties(
            eqm_props_dict, self.__system, path_output=path_output, dpi=dpi,
        )
            
        return self

    def run(self, path_output='.'):
        pass

  
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', 
                        type=str, 
                        default='./',
                        help='Path (output of predictions)')
    args = parser.parse_args()

    pred = PolymlpRepositoryGeneration(path_data=args.path_data)
    pred.run_mlp_distribution()
    pred.run_eos()

