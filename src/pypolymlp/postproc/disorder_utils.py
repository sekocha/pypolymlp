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


def set_element_map(params: PolymlpParams, occupancy: tuple):
    """Set map between elements and atom types."""
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

    return map_element_to_type


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


def generate_substitutional_structures(
    lattice: PolymlpStructure,
    occupancy: list,
    map_element_to_type: dict,
    n_samples: int = 500,
):
    """Generate random substitutional structures."""

    if lattice is None:
        raise RuntimeError("Lattice not found.")

    structures, atom_orders = [], []
    for i in range(n_samples):
        replace_ids = defaultdict(list)
        atom_begin = 0
        for occ, n in zip(occupancy, lattice.n_atoms, strict=True):
            atom_end = atom_begin + n
            cand = range(atom_begin, atom_end)
            for ele, prob in occ:
                n_replace = int(round(n * prob))
                itype = map_element_to_type[ele]

                ele_str = ele[0] if len(ele) == 2 else ele
                if len(cand) == n_replace:
                    replace_ids[(ele_str, itype)].extend(cand)
                else:
                    samples = random.sample(cand, n_replace)
                    replace_ids[(ele_str, itype)].extend(samples)
                    cand = list(set(cand) - set(replace_ids[(ele_str, itype)]))
            atom_begin = atom_end

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


def eval_substitutional_structures(
    calc: PypolymlpCalc,
    lattice: PolymlpStructure,
    occupancy: list,
    map_element_to_type: dict,
    n_samples: int = 500,
):
    """Evaluate properties of random substitutional structures."""
    subs, atom_orders = generate_substitutional_structures(
        lattice,
        occupancy,
        map_element_to_type,
        n_samples=n_samples,
    )
    energies, forces_sorted_order, stresses = calc.eval(subs)
    forces = []
    for f, ids in zip(forces_sorted_order, atom_orders, strict=True):
        f_reordered = np.zeros(f.shape)
        f_reordered[:, ids] = f
        forces.append(f_reordered)
    return energies, forces, stresses
