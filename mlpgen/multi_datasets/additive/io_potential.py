#!/usr/bin/env python
import numpy as np

from pypolymlp.mlpgen.io_potential import save_mlp_lammps

def save_multiple_mlp_lammps(multiple_params_dicts, 
                             cumulative_n_features,
                             coeffs, 
                             scales):

    for i, params_dict in enumerate(multiple_params_dicts):
        if i == 0:
            begin, end = 0, cumulative_n_features[0]
        else:
            begin, end = cumulative_n_features[i-1], cumulative_n_features[i]

        save_mlp_lammps(params_dict,
                        coeffs[begin:end],
                        scales[begin:end],
                        filename='polymlp.lammps.'+str(i+1))


