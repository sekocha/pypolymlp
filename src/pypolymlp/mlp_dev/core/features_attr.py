"""Functions for obtaining feature attributes."""

from collections import defaultdict
from typing import Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.cxx.lib import libmlpcpp


def _get_num_features(params: PolymlpParams):
    """Return number of features."""
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)
    n_fearures = len(features_attr) + len(polynomial_attr)
    return n_fearures


def get_num_features(params: Union[PolymlpParams, list[PolymlpParams]]):
    """Return number of features."""
    if isinstance(params, list):
        n_features = 0
        for i, p in enumerate(params):
            n_features += _get_num_features(p)
        return n_features
    return _get_num_features(params)


def get_features_attr(params: PolymlpParams, element_swap: bool = False):
    """Get feature attributes."""
    params.element_swap = element_swap
    obj = libmlpcpp.FeaturesAttr(params.as_dict())

    type_pairs = obj.get_type_pairs()
    atomtype_pair_dict = defaultdict(list)
    for type1, tps in enumerate(type_pairs):
        for type2, tp in enumerate(tps):
            atomtype_pair_dict[tp].append([type1, type2])

    radial_ids = obj.get_radial_ids()
    tcomb_ids = obj.get_tcomb_ids()
    polynomial_attr = obj.get_polynomial_ids()
    if params.model.feature_type == "pair":
        features_attr = list(zip(radial_ids, tcomb_ids))
    elif params.model.feature_type == "gtinv":
        gtinv_ids = obj.get_gtinv_ids()
        features_attr = list(zip(radial_ids, gtinv_ids, tcomb_ids))

    return features_attr, polynomial_attr, atomtype_pair_dict


def _write_polymlp_params_yaml(
    params: PolymlpParams,
    filename: str = "polymlp_params.yaml",
):
    """Save feature attributes to yaml file."""
    np.set_printoptions(legacy="1.21")
    f = open(filename, "w")
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)

    elements = np.array(params.elements)
    print("radial_params:", file=f)
    for i, p in enumerate(params.model.pair_params):
        print("- radial_id: ", i, file=f)
        print("  params:    ", list(p), file=f)
        print("", file=f)
    print("", file=f)

    print("atomtype_pairs:", file=f)
    for k, v in atomtype_pair_dict.items():
        print("- pair_id:       ", k, file=f)
        print("  pair_elements: ", file=f)
        for v1 in v:
            print("  - ", list(elements[v1]), file=f)
        print("", file=f)
    print("", file=f)

    print("features:", file=f)
    seq_id = 0
    if params.model.feature_type == "pair":
        for i, attr in enumerate(features_attr):
            print("- id:               ", seq_id, file=f)
            print("  feature_id:       ", i, file=f)
            print("  radial_id:        ", attr[0], file=f)
            print("  atomtype_pair_ids:", attr[1], file=f)
            print("", file=f)
            seq_id += 1
        print("", file=f)

    elif params.model.feature_type == "gtinv":
        gtinv = params.model.gtinv
        for i, attr in enumerate(features_attr):
            print("- id:               ", seq_id, file=f)
            print("  feature_id:       ", i, file=f)
            print("  radial_id:        ", attr[0], file=f)
            print("  gtinv_id:         ", attr[1], file=f)
            print(
                "  l_combination:    ",
                gtinv.l_comb[attr[1]],
                file=f,
            )
            print("  atomtype_pair_ids:", attr[2], file=f)
            print("", file=f)
            seq_id += 1
        print("", file=f)

    if len(polynomial_attr) > 0:
        print("polynomial_features:", file=f)
        for i, attr in enumerate(polynomial_attr):
            print("- id:          ", seq_id, file=f)
            print("  feature_ids: ", attr, file=f)
            print("", file=f)
            seq_id += 1

    f.close()
    return seq_id


def write_polymlp_params_yaml(
    self,
    params: Union[PolymlpParams, list[PolymlpParams]],
    filename: str = "polymlp_params.yaml",
):
    """Write polymlp_params.yaml"""
    np.set_printoptions(legacy="1.21")
    if isinstance(params, list):
        self._n_features = 0
        for i, p in enumerate(params):
            filename = "polymlp_params" + str(i + 1) + ".yaml"
            self._n_features += _write_polymlp_params_yaml(p, filename=filename)
    else:
        self._n_features = _write_polymlp_params_yaml(params, filename=filename)
    return self._n_features


# def print_params(
#     params: Union[PolymlpParams, list[PolymlpParams]],
#     common_params: PolymlpParams,
# ):
#     """Print parameters."""
#     if self._hybrid:
#         print("priority_input:", self._priority_infile, flush=True)
#
#     params = self.common_params
#     print("parameters:", flush=True)
#     print("  n_types:       ", params.n_type, flush=True)
#     print("  elements:      ", params.elements, flush=True)
#     print("  element_order: ", params.element_order, flush=True)
#     print("  atomic_energy: ", params.atomic_energy, flush=True)
#     print("  include_force: ", bool(params.include_force), flush=True)
#     print("  include_stress:", bool(params.include_stress), flush=True)
#
#     if self.is_multiple_datasets:
#         print("  train_data:", flush=True)
#         for v in params.dft_train:
#             print("  -", v.location, flush=True)
#         print("  test_data:", flush=True)
#         for v in params.dft_test:
#             print("  -", v.location, flush=True)
#     else:
#         if params.dataset_type == "phono3py":
#             print("  train_data:", flush=True)
#             print("  -", params.dft_train["phono3py_yaml"], flush=True)
#             print("  test_data:", flush=True)
#             print("  -", params.dft_test["phono3py_yaml"], flush=True)
#         else:
#             pass
#
#     if isinstance(self.params, PolymlpParams):
#         params = [self.params]
#     else:
#         params = self.params
#     for i, p in enumerate(params):
#         print("model_" + str(i + 1) + ":", flush=True)
#         print("  cutoff:      ", p.model.cutoff, flush=True)
#         print("  model_type:  ", p.model.model_type, flush=True)
#         print("  max_p:       ", p.model.max_p, flush=True)
#         print("  n_gaussians: ", len(p.model.pair_params), flush=True)
#         print("  feature_type:", p.model.feature_type, flush=True)
#         if p.model.feature_type == "gtinv":
#             orders = [i for i in range(2, p.model.gtinv.order + 1)]
#             print("  max_l:       ", p.model.gtinv.max_l, end=" ", flush=True)
#             print("for order =", orders, flush=True)
