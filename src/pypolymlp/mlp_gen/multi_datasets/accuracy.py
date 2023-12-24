#!/usr/bin/env python
import numpy as np
import copy

from pypolymlp.mlp_gen.features import Features
   
def compute_predictions(params_dict, dft_dict, coeffs, scales):

    params_include_force = copy.copy(params_dict['include_force'])
    params_dict['include_force'] = dft_dict['include_force']
    features = Features(params_dict, dft_dict['structures'])
    params_dict['include_force'] = params_include_force
    x = features.get_x()
    indices = features.get_first_indices()[0]

    coeffs_rescale = coeffs / scales
    predictions = np.dot(x, coeffs_rescale)
    weights = np.ones(len(predictions))

    return predictions, weights, indices

