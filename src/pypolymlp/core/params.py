"""Class for input parameters including hybrid polymlps."""

import copy
from typing import Optional, Union

from pypolymlp.core.data_format import PolymlpParamsSingle


def _get_variable_with_max_length(
    multiple_params: list[PolymlpParamsSingle], key: str
) -> list:
    """Select variable with max length."""
    array = []
    for single in multiple_params:
        single_dict = single.as_dict()
        if len(single_dict[key]) > len(array):
            array = single_dict[key]
    return array


def set_common_params(
    multiple_params: list[PolymlpParamsSingle],
) -> PolymlpParamsSingle:
    """Set common parameters of multiple PolymlpParams."""
    keys = set()
    for single in multiple_params:
        for k in single.as_dict().keys():
            keys.add(k)

    common_params = copy.copy(multiple_params[0])
    n_type = max([single.n_type for single in multiple_params])
    elements = _get_variable_with_max_length(multiple_params, "elements")
    atom_e = _get_variable_with_max_length(multiple_params, "atomic_energy")

    bool_element_order = [
        single.element_order for single in multiple_params
    ] is not None
    element_order = elements if bool_element_order else None

    common_params.n_type = n_type
    common_params.elements = elements
    common_params.element_order = element_order
    common_params.atomic_energy = atom_e
    return common_params


# def _set_unique_types(params: PolymlpParams):
#     """Set type indices for hybrid models."""
#     if not params.is_hybrid:
#         return params
#
#     n_type = params.n_type
#     elements = params.elements
#     for single in params:
#         single.elements = sorted(single.elements, key=lambda x: elements.index(x))
#
#     for single in params:
#         if single.n_type == n_type:
#             single.type_full = True
#             single.type_indices = list(range(n_type))
#         else:
#             single.type_full = False
#             single.type_indices = [elements.index(ele) for ele in single.elements]
#     return params


class PolymlpParams:
    """Class for input parameters including hybrid polymlps."""

    def __init__(self, params: Optional[Union[list, PolymlpParamsSingle]] = None):
        """Init method."""
        if params is None:
            self._params = []
        elif isinstance(params, PolymlpParamsSingle):
            self._params = [params]
        else:
            self._params = params
        self._common_params = self._set_common_params()

    def __iter__(self):
        """Iter method."""
        return iter(self._params)

    def __getitem__(self, index: int):
        """Getitem method."""
        return self._params[index]

    def __setitem__(self, index: int, value: PolymlpParamsSingle):
        """Setitem method."""
        self._params[index] = value

    def __len__(self):
        """Len method."""
        return len(self._params)

    def append(self, params: PolymlpParamsSingle):
        """Append parameters."""
        self._params.append(params)
        self._common_params = self._set_common_params()

    def _set_common_params(self):
        """Set common parameters in hybrid model."""
        if len(self._params) == 0:
            self._common_params = None
        elif len(self._params) == 1:
            self._common_params = self._params[0]
        else:
            self._common_params = set_common_params(self._params)
        self._set_unique_types()
        return self._common_params

    def _set_unique_types(self):
        """Set unique type indices for hybrid model."""
        if not self.is_hybrid:
            return self

        n_type = self.n_type
        elements = self.elements
        for single in self._params:
            single.elements = sorted(single.elements, key=lambda x: elements.index(x))

        for single in self._params:
            if single.n_type == n_type:
                single.type_full = True
                single.type_indices = list(range(n_type))
            else:
                single.type_full = False
                single.type_indices = [elements.index(ele) for ele in single.elements]
        return self

    @property
    def params(self):
        """Return parameters or parameter list."""
        if len(self) > 1:
            return self._params
        elif len(self) == 1:
            return self._params[0]
        return None

    @property
    def n_type(self):
        """Return number of atom types."""
        return self._common_params.n_type

    @property
    def elements(self):
        """Return element strings."""
        return self._common_params.elements

    @property
    def element_order(self):
        """Return element order."""
        return self._common_params.element_order

    @property
    def atomic_energy(self):
        """Return atomic energies."""
        return self._common_params.atomic_energy

    @property
    def include_force(self):
        """Return whether forces are included or not."""
        return bool(self._common_params.include_force)

    @include_force.setter
    def include_force(self, include: bool):
        """Setter of include_force."""
        self._common_params.include_force = include

    @property
    def include_stress(self):
        """Return whether stresses are included or not."""
        return bool(self._common_params.include_stress)

    @include_stress.setter
    def include_stress(self, include: bool):
        """Setter of include_stress."""
        self._common_params.include_stress = include

    @property
    def dataset_type(self):
        """Return dataset type."""
        return bool(self._common_params.dataset_type)

    @property
    def alphas(self):
        """Return whether stresses are included or not."""
        return self._common_params.alphas

    @property
    def is_hybrid(self):
        """Return whether model is hybrid one or not."""
        return len(self._params) > 1

    def print_params(self):
        """Print parameters."""
        # print("priority_input:", common_params.priority_infile, flush=True)
        print("parameters:", flush=True)
        print("  n_types:         ", self.n_type, flush=True)
        print("  elements:        ", self.elements, flush=True)
        print("  element_order:   ", self.element_order, flush=True)
        print("  atomic_energy_eV:", self.atomic_energy, flush=True)
        print("  include_force:   ", self.include_force, flush=True)
        print("  include_stress:  ", self.include_stress, flush=True)

        for i, p in enumerate(self._params):
            print("model_" + str(i + 1) + ":", flush=True)
            print("  cutoff:      ", p.model.cutoff, flush=True)
            print("  model_type:  ", p.model.model_type, flush=True)
            print("  max_p:       ", p.model.max_p, flush=True)
            print("  n_gaussians: ", len(p.model.pair_params), flush=True)
            print("  feature_type:", p.model.feature_type, flush=True)
            if p.model.feature_type == "gtinv":
                orders = [i for i in range(2, p.model.gtinv.order + 1)]
                print("  max_l:       ", p.model.gtinv.max_l, end=" ", flush=True)
                print("for order =", orders, flush=True)
