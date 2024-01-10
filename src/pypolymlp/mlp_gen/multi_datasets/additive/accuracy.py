#!/usr/bin/env python
import numpy as np

from pypolymlp.mlp_gen.multi_datasets.additive.features import Features

def compute_predictions(multiple_params_dicts, dft_dict, coeffs, scales):

    dft_dict_tmp = dict({'set1': dft_dict})
    features = Features(multiple_params_dicts, dft_dict_tmp)

    x = features.get_x()
    indices = features.get_first_indices()[0]

    coeffs_rescale = coeffs / scales
    predictions = np.dot(x, coeffs_rescale)
    weights = np.ones(len(predictions))

    return predictions, weights, indices


