"""Utility functions for developing disordered model."""

import copy
import random
from collections import defaultdict

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.params import PolymlpParams
from pypolymlp.utils.structure_utils import sort_wrt_types


def _check_occupancy(params: PolymlpParams, occupancy: tuple):
    """Check occupancy format."""
    for occ in occupancy:
        if not np.isclose(sum([v for _, v in occ]), 1.0):
            raise RuntimeError("Sum of occupancy != 1.0")
        for ele, _ in occ:
            if isinstance(ele, str):
                if ele not in params.elements:
                    raise RuntimeError("Element " + ele + " not found in polymlp.")
            elif len(ele) == 2:
                if ele[0] not in params.elements:
                    raise RuntimeError("Element " + ele[0] + " not found in polymlp.")
                ele_id = list(params.elements).index(ele[0])
                if not params.enable_spins[ele_id]:
                    raise RuntimeError("Element " + ele[0] + " has no spin.")
            else:
                raise RuntimeError("Broken occupancy format.")


def set_full_occupancy(params: PolymlpParams, occupancy: tuple):
    """Set occupancy list for generating disorder structures."""
    _check_occupancy(params, occupancy)

    map_element_to_type = dict()
    unique_elements = list(dict.fromkeys(params.elements))
    itype = 0
    for ele in unique_elements:
        n_match = np.count_nonzero(np.array(params.elements) == ele)
        if n_match == 1:
            map_element_to_type[ele] = itype
            itype += 1
        else:
            map_element_to_type[(ele, 0)] = itype
            itype += 1
            map_element_to_type[(ele, 1)] = itype
            itype += 1

    ele_all = set([ele for occ in occupancy for ele, _ in occ])
    for ele2 in map_element_to_type.keys():
        assert ele2 in ele_all

    occupancy_full = []
    for occ in occupancy:
        occ_sub = []
        for ele, prob in occ:
            if isinstance(ele, str):
                ele_str = ele
            elif isinstance(ele, tuple):
                ele_str = ele[0]
            occ_sub.append((ele_str, map_element_to_type[ele], prob))
        occupancy_full.append(occ_sub)

    return occupancy_full


def _generate_substitutional_indices(lattice: PolymlpStructure, occupancy: list):
    """Generate a set of substitutional indices for a substitutional structure."""
    replace_ids = defaultdict(list)
    atom_begin = 0
    for occ, n in zip(occupancy, lattice.n_atoms, strict=True):
        atom_end = atom_begin + n
        cand = range(atom_begin, atom_end)
        for ele, itype, prob in occ:
            key = (ele, itype)
            n_rep = int(round(n * prob))
            samples = cand if len(cand) == n_rep else random.sample(cand, n_rep)
            replace_ids[key].extend(samples)
            cand = list(set(cand) - set(replace_ids[key]))
        atom_begin = atom_end
    return replace_ids


def generate_substitutional_structures(
    lattice: PolymlpStructure,
    occupancy: list,
    n_samples: int = 500,
):
    """Generate random substitutional structures."""

    if lattice is None:
        raise RuntimeError("Lattice not found.")

    structures, atom_orders = [], []
    for i in range(n_samples):
        replace_ids = _generate_substitutional_indices(lattice, occupancy)

        st = copy.deepcopy(lattice)
        st.elements = np.array(st.elements)
        st.types = np.array(st.types)
        for (ele, itype), rep_ids in replace_ids.items():
            st.elements[rep_ids] = ele
            st.types[rep_ids] = itype

        st, ids = sort_wrt_types(st, return_ids=True)
        structures.append(st)
        atom_orders.append(ids)
    return structures, np.array(atom_orders)


def _reorder(array: np.ndarray, order: np.ndarray):
    """Reorder array with respect to the order of original array."""
    array_reordered = np.zeros(array.shape)
    array_reordered[:, order] = array
    return array_reordered


def eval_substitutional_structures(
    calc: PypolymlpCalc,
    lattice: PolymlpStructure,
    occupancy: list,
    n_samples: int = 500,
):
    """Evaluate properties of random substitutional structures."""
    subs, atom_orders = generate_substitutional_structures(
        lattice,
        occupancy,
        n_samples=n_samples,
    )
    energies, forces_sorted_order, stresses = calc.eval(subs)
    zip_array = zip(forces_sorted_order, atom_orders, strict=True)
    forces = [_reorder(f, ids) for f, ids in zip_array]
    return energies, forces, stresses


# def _check_params(params: PolymlpParams):
#     """Check constraints for parameters."""
#     if not params.type_full:
#         raise RuntimeError("type_full must be True")
#
#     uniq_ids = set()
#     for ids in params.model.pair_params_conditional.values():
#         uniq_ids.add(tuple(ids))
#
#     if len(uniq_ids) != 1:
#         raise RuntimeError("All pair_params_conditional must be the same.")
#
#     return True
#
#
# def _generate_disorder_params(params: PolymlpParams, occupancy: tuple):
#     """Check occupancy format."""
#     _check_params(params)
#
#     params_rand = copy.deepcopy(params)
#     map_mass = dict(zip(params.elements, params.mass))
#
#     elements_rand, mass_rand = [], []
#     type_group = []
#     for occ in occupancy:
#         if not np.isclose(sum([v for _, v in occ]), 1.0):
#             raise RuntimeError("Sum of occupancy != 1.0")
#
#         mass, type_tmp = 0.0, []
#         for ele, comp in occ:
#             if ele not in params.elements:
#                 raise RuntimeError("Element", ele, "not found in polymlp.")
#
#             mass += map_mass[ele] * comp
#             type_tmp.append(params.elements.index(ele))
#
#         elements_rand.append(occ[0][0])
#         mass_rand.append(mass)
#         type_group.append(type_tmp)
#
#     params_rand.n_type = len(occupancy)
#     params_rand.elements = elements_rand
#     params_rand.element_order = elements_rand
#     params_rand.mass = mass_rand
#
#     # TODO: Modify type_pairs and type_indices
#     #       Use type_group
#     params_rand.type_full = True
#     params_rand.type_indices = list(range(params_rand.n_type))
#
#     occupancy_type = [
#         [(params.elements.index(ele), comp) for ele, comp in occ] for occ in occupancy
#     ]
#     return params_rand, occupancy_type
