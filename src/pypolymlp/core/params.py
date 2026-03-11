"""Class for input parameters including hybrid polymlps."""

import copy
from typing import Optional, Union

from pypolymlp.core.data_format import PolymlpParams


def _get_variable_with_max_length(
    multiple_params: list[PolymlpParams], key: str
) -> list:
    """Select variable with max length."""
    array = []
    for single in multiple_params:
        single_dict = single.as_dict()
        if len(single_dict[key]) > len(array):
            array = single_dict[key]
    return array


def set_common_params(multiple_params: list[PolymlpParams]) -> PolymlpParams:
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


class PolymlpParamsList:
    """Class for input parameters including hybrid polymlps."""

    def __init__(self, params: Optional[Union[list, PolymlpParams]] = None):
        """Init method."""
        if params is None:
            self._params = []
        elif isinstance(params, PolymlpParams):
            self._params = [params]
        else:
            self._params = params

        self._common_params = None

    def __iter__(self):
        """Iter method."""
        return iter(self._params)

    def __len__(self):
        """Len method."""
        return len(self._params)

    def append(self, params: PolymlpParams):
        """Append parameters."""
        self._params.append(params)
        self._common_params = None

    @property
    def params(self):
        """Return parameters or parameter list."""
        if len(self) > 1:
            return self._params
        elif len(self) == 1:
            return self._params[0]
        return None

    @property
    def common_params(self):
        """Return common parameters for hybrid model."""
        if len(self) > 1:
            if self._common_params is None:
                self._common_params = set_common_params(self._params)
        elif len(self) == 1:
            self._common_params = self._params[0]
        return self._common_params
