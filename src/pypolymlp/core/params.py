"""Class for input parameters including hybrid polymlps."""

import copy
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParamsSingle


def _get_variable_with_max_length(
    multiple_params: list[PolymlpParamsSingle], key: str
) -> list:
    """Select variable with max length."""
    array = []
    for single in multiple_params:
        target = getattr(single, key)
        if target is None:
            continue
        if len(target) > len(array):
            array = target
    return array


def set_common_params(
    multiple_params: list[PolymlpParamsSingle],
) -> PolymlpParamsSingle:
    """Set common parameters of multiple PolymlpParams."""
    common_params = copy.copy(multiple_params[0])
    n_type = max([single.n_type for single in multiple_params])
    elements = _get_variable_with_max_length(multiple_params, "elements")
    atom_e = _get_variable_with_max_length(multiple_params, "atomic_energy")
    enable_spins = _get_variable_with_max_length(multiple_params, "enable_spins")

    bool_element_order = [
        single.element_order for single in multiple_params
    ] is not None
    element_order = elements if bool_element_order else None

    common_params.n_type = n_type
    common_params.elements = tuple(elements)
    common_params.element_order = tuple(element_order)
    common_params.atomic_energy = tuple(atom_e)
    if len(enable_spins) == 0:
        common_params.enable_spins = None
    else:
        common_params.enable_spins = tuple(enable_spins)
    return common_params


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
        self._check_dataset_type()
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
        """Return parameters or parameter list.

        Return
        ------
        Parameters. If single model is defined, parameters in PolymlpParamsSingle
        are returned. In the case of hybrid model, list of parameters
        in PolymlpParamsSingle will be returned.
        """
        if len(self) > 1:
            return self._params
        elif len(self) == 1:
            return self._params[0]
        return None

    @params.setter
    def params(
        self,
        p: Union[PolymlpParamsSingle, list[PolymlpParamsSingle]],
    ):
        """Setter of parameters."""
        if p is None:
            self._params = []
        elif isinstance(p, PolymlpParamsSingle):
            self._params = [p]
        elif isinstance(p, (list, tuple, np.ndarray)):
            self._params = list(p)
        else:
            raise RuntimeError("Inappropriate input for parameters.")
        self._common_params = self._set_common_params()

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
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")

        self._common_params.include_force = include
        for p in self._params:
            p.include_force = include
        self._check_dataset_type()

    @property
    def include_stress(self):
        """Return whether stresses are included or not."""
        return bool(self._common_params.include_stress)

    @include_stress.setter
    def include_stress(self, include: bool):
        """Setter of include_stress."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.include_stress = include
        for p in self._params:
            p.include_stress = include
        self._check_dataset_type()

    @property
    def enable_spins(self):
        """Return whether spins are included or not."""
        return self._common_params.enable_spins

    @enable_spins.setter
    def enable_spins(self, spins: tuple[bool]):
        """Setter of include_spin."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.enable_spins = spins
        for p in self._params:
            p.enable_spins = spins
        self._check_dataset_type()

    @property
    def dataset_type(self):
        """Return dataset type."""
        return self._common_params.dataset_type

    @dataset_type.setter
    def dataset_type(self, dtype: str):
        """Setter of dataset type."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.dataset_type = dtype
        for p in self._params:
            p.dataset_type = dtype
        self._check_dataset_type()

    @property
    def temperature(self):
        """Return temperature."""
        return self._common_params.temperature

    @temperature.setter
    def temperature(self, temp: float):
        """Setter of temperature."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.temperature = temp
        for p in self._params:
            p.temperature = temp

    @property
    def electron_property(self):
        """Return target electronic property."""
        return self._common_params.electron_property

    @electron_property.setter
    def electron_property(self, prop: str):
        """Setter of target electronic property."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.electron_property = prop
        for p in self._params:
            p.electron_property = prop

    @property
    def element_swap(self):
        """Return element_swap."""
        return bool(self._common_params.element_swap)

    @element_swap.setter
    def element_swap(self, es: str):
        """Setter of element_swap."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.element_swap = es
        for p in self._params:
            p.element_swap = es

    @property
    def print_memory(self):
        """Return print_memory."""
        return bool(self._common_params.print_memory)

    @print_memory.setter
    def print_memory(self, pm: str):
        """Setter of print_memory."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.print_memory = pm
        for p in self._params:
            p.print_memory = pm

    @property
    def regression_alpha(self):
        """Return power indices of alpha penalty values."""
        return self._common_params.regression_alpha

    @regression_alpha.setter
    def regression_alpha(self, a: str):
        """Setter of print_memory."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.regression_alpha = a
        for p in self._params:
            p.regression_alpha = a

    @property
    def alphas(self):
        """Return alpha penalty values."""
        return self._common_params.alphas

    @alphas.setter
    def alphas(self, a: str):
        """Setter of print_memory."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")
        self._common_params.alphas = a
        for p in self._params:
            p.alphas = a

    @property
    def is_hybrid(self):
        """Return whether model is hybrid one or not."""
        return len(self._params) > 1

    def as_dict(self):
        """Convert parameters to dictionary or list of dictionary."""
        if self.is_hybrid:
            return [p.as_dict() for p in self._params]
        return self._params[0].as_dict()

    def print_params(self):
        """Print parameters."""
        print("parameters:", flush=True)
        print("  n_types:           ", self.n_type, flush=True)
        print("  elements:          ", self.elements, flush=True)
        print("  element_order:     ", self.element_order, flush=True)
        print("  atomic_energy (eV):", self.atomic_energy, flush=True)
        print("  include_force:     ", self.include_force, flush=True)
        print("  include_stress:    ", self.include_stress, flush=True)
        print("  enable_spins:      ", self.enable_spins, flush=True)

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

    def as_hybrid_model(self):
        """Make a hybrid model used for tests."""
        if len(self._params) > 1:
            raise RuntimeError("Hybrid model not required to convert.")
        return PolymlpParams([self._params[0], self._params[0]])

    def _check_dataset_type(self):
        """Check whether dataset type is available for given parameters."""
        if self._common_params is None:
            return

        if self.dataset_type in ("phono3py", "sscha", "openmx"):
            if self.include_stress:
                raise RuntimeError(
                    "Include_stress not supported for given dataset type."
                )
            if self.enable_spins is not None:
                raise RuntimeError("Spin not supported for given dataset type.")

        elif self.dataset_type == "electron":
            if self.include_force:
                raise RuntimeError(
                    "Include_force not supported for given dataset type."
                )
            if self.include_stress:
                raise RuntimeError(
                    "Include_stress not supported for given dataset type."
                )
            if self.enable_spins is not None:
                raise RuntimeError("Spin not supported in phono3py dataset.")
