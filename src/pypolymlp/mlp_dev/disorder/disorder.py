"""Generator of polymlp for disordered model."""

import copy
import itertools

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.io_polymlp import load_mlp, save_mlp
from pypolymlp.mlp_dev.core.features_attr import get_features_attr


def _check_params(params: PolymlpParams):
    """Check constraints for parameters."""
    if not params.type_full:
        raise RuntimeError("type_full must be True")

    uniq_ids = set()
    for ids in params.model.pair_params_conditional.values():
        uniq_ids.add(tuple(ids))

    if len(uniq_ids) != 1:
        raise RuntimeError("All pair_params_conditional must be the same.")

    return True


def _generate_disorder_params(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    _check_params(params)

    params_rand = copy.deepcopy(params)
    map_mass = dict(zip(params.elements, params.mass))

    elements_rand, mass_rand = [], []
    type_group = []
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")

        mass, type_tmp = 0.0, []
        for ele, comp in occ:
            if ele not in params.elements:
                raise RuntimeError("Element", ele, "not found in polymlp.")

            mass += map_mass[ele] * comp
            type_tmp.append(params.elements.index(ele))

        elements_rand.append(occ[0][0])
        mass_rand.append(mass)
        type_group.append(type_tmp)

    params_rand.n_type = len(occupancy)
    params_rand.elements = elements_rand
    params_rand.mass = mass_rand

    # TODO: Modify type_pairs and type_indices
    # Use type_group

    occupancy_type = [
        [(params.elements.index(ele), comp) for ele, comp in occ] for occ in occupancy
    ]
    return params_rand, occupancy_type


def set_mapping_original_mlp(params: PolymlpParams, coeffs: np.ndarray):
    """Set mapping of original MLP."""
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params)

    seq_id = 0
    map_feature = dict()
    if params.model.feature_type == "gtinv":
        gtinv = params.model.gtinv
        features = []
        for attr in features_attr:
            # attr: (radial_id, gtinv_id, type_pair_id)
            rad_id = attr[0]
            lc = tuple(gtinv.l_comb[attr[1]])
            tp = tuple([tuple(atomtype_pair_dict[i][0]) for i in attr[2]])
            key = (rad_id, lc, tp)
            features.append(key)
            map_feature[key] = coeffs[seq_id]
            seq_id += 1
    else:
        pass

    map_polynomial = dict()
    if len(polynomial_attr) > 0:
        for attr in polynomial_attr:
            key = tuple([features[i] for i in attr])
            map_polynomial[key] = coeffs[seq_id]
            seq_id += 1

    return map_feature, map_polynomial


def find_central_types(type_pairs: list):
    """Find atom types of central atoms for a combination of type pairs."""
    common_type = set(type_pairs[0])
    for tp in type_pairs[1:]:
        common_type = common_type & set(tp)

    central = dict()
    for t1 in common_type:
        neighbors = []
        for tp in type_pairs:
            neigh = tp[1] if tp[0] == t1 else tp[0]
            neighbors.append(neigh)
        central[t1] = neighbors
    return central


def generate_disorder_mlp_pair(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP with pair invariants for disorder model."""
    map_feature, map_polynomial = set_mapping_original_mlp(params, coeffs)
    params_rand, occupancy_type = _generate_disorder_params(params, occupancy)
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params_rand)

    # TODO: function for pair
    return None, None


def generate_disorder_mlp_gtinv(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP with polynomial invariants for disorder model."""
    map_feature, map_polynomial = set_mapping_original_mlp(params, coeffs)
    params_rand, occupancy_type = _generate_disorder_params(params, occupancy)
    features_attr, polynomial_attr, atomtype_pair_dict = get_features_attr(params_rand)

    gtinv = params_rand.model.gtinv
    coeffs_rand, features = [], []
    for attr in features_attr:
        rad_id = attr[0]
        lc = tuple(gtinv.l_comb[attr[1]])
        tp = [tuple(atomtype_pair_dict[i][0]) for i in attr[2]]
        features.append((rad_id, lc, tp))

        central = find_central_types(tp)
        coeff = 0.0
        for ctype, neigh_types in central.items():
            site_occupancies = [occupancy_type[ctype]]
            for i in neigh_types:
                site_occupancies.append(occupancy_type[i])

            occ_comb = itertools.product(*site_occupancies)
            for occ in occ_comb:
                prob = np.prod([p for t, p in occ])
                center_type = occ[0][0]
                tp = [tuple(sorted([center_type, t])) for t, p in occ[1:]]
                lc_tp = sorted([(l, t) for l, t in zip(lc, tp)])
                tp = tuple([t for l, t in lc_tp])
                key = (rad_id, lc, tp)
                coeff += prob * map_feature[key]

        coeff /= len(central)
        coeffs_rand.append(coeff)

    if len(polynomial_attr) > 0:
        for attr in polynomial_attr:
            rad_id_array = [features[i][0] for i in attr]
            lc_array = [tuple([l for l in features[i][1]]) for i in attr]
            tp = [t for i in attr for t in features[i][2]]
            central = find_central_types(tp)

            coeff = 0.0
            for ctype, neigh_types in central.items():
                site_occupancies = [occupancy_type[ctype]]
                for i in neigh_types:
                    site_occupancies.append(occupancy_type[i])
                occ_comb = itertools.product(*site_occupancies)
                for occ in occ_comb:
                    prob = np.prod([p for t, p in occ])
                    center_type = occ[0][0]
                    tp = [tuple(sorted([center_type, t])) for t, p in occ]

                    begin, key = 0, []
                    for rad, lc in zip(rad_id_array, lc_array):
                        end = begin + len(lc)
                        lc_tp = sorted([(l, t) for l, t in zip(lc, tp[begin:end])])
                        tp_key = tuple([t for l, t in lc_tp])
                        key.append((rad, lc, tp_key))
                        begin = end
                    key = tuple(sorted(key))

                    coeff += prob * map_polynomial[key]

            coeff /= len(central)
            coeffs_rand.append(coeff)

    return params_rand, np.array(coeffs_rand)


def generate_disorder_mlp(
    params: PolymlpParams,
    coeffs: np.ndarray,
    occupancy: tuple,
):
    """Generate MLP for disorder model."""
    if params.model.feature_type == "pair":
        return generate_disorder_mlp_pair(params, coeffs, occupancy)
    return generate_disorder_mlp_gtinv(params, coeffs, occupancy)


# params, coeffs = load_mlp("polymlp.yaml")
# occupancy = [[("La", 0.75), ("Cu", 0.25)], [("O", 1.0)], [("Te", 1.0)]]

params, coeffs = load_mlp("polymlp.yaml")
# params, coeffs = load_mlp("polymlp.lammps")
occupancy = [[("Ag", 0.5), ("Au", 0.5)]]

params_rand, coeffs_rand = generate_disorder_mlp(params, coeffs, occupancy)

save_mlp(
    params_rand,
    coeffs_rand,
    scales=np.ones(coeffs_rand.shape),
    filename="polymlp.yaml.disorder",
)
