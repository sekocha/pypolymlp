"""Class for input parameters including hybrid polymlps."""

import copy
from typing import Any, Union

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

    common_params.n_type = n_type
    common_params.elements = tuple(elements)
    if len(atom_e) == 0:
        common_params.atomic_energy = None
    else:
        common_params.atomic_energy = tuple(atom_e)
    if len(enable_spins) == 0:
        common_params.enable_spins = None
    else:
        common_params.enable_spins = tuple(enable_spins)
    return common_params


class PolymlpParams:
    """Class for input parameters including hybrid polymlps."""

    def __init__(self, params: Union[list, PolymlpParamsSingle]):
        """Init method."""
        if isinstance(params, PolymlpParamsSingle):
            self._params = [params]
        elif isinstance(params, (list, tuple, np.ndarray)):
            self._params = params
        else:
            raise RuntimeError("Initialize failed. Inappropriate instance.")

        self._initialize()

    def _initialize(self):
        """Initialize variables.

        This initialization is needed if parameters are changed.
        """
        self._set_common_params()
        self._check_errors()
        self._set_tags()
        self._set_unique_types()

    def __iter__(self):
        """Iter method."""
        return iter(self._params)

    def __getitem__(self, index: int):
        """Getitem method."""
        return self._params[index]

    def __setitem__(self, index: int, value: PolymlpParamsSingle):
        """Setitem method."""
        self._params[index] = value
        self._initialize()

    def __len__(self):
        """Len method."""
        return len(self._params)

    def append(self, params: PolymlpParamsSingle):
        """Append parameters."""
        self._params.append(params)
        self._initialize()

    def _check_errors(self):
        """Check errors in parameters."""
        if len(self._params) == 1:
            return self

        n_none = 0
        for p in self._params:
            if p.enable_spins is None:
                n_none += 1
            elif np.all(np.array(p.enable_spins) == False):
                n_none += 1

        if n_none == len(self._params):
            return self

        if n_none != 0 and n_none != len(self._params):
            raise RuntimeError(
                "enable_spins not consistent across all components of hybrid model."
            )

        combs = [
            (ele, spin)
            for p in self.params
            for ele, spin in zip(p.elements, p.enable_spins)
        ]
        if len(set(combs)) != len(np.unique(self.elements)):
            raise RuntimeError("Same values of enable_spins not given.")

    def _set_common_params(self):
        """Set common parameters in hybrid model."""
        if len(self._params) == 1:
            self._common_params = self._params[0]
        else:
            self._common_params = set_common_params(self._params)
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

    def setter_common_property(self, name: str, value: Any):
        """Setter of common target property."""
        if self._common_params is None:
            raise RuntimeError("Parameters not defined.")

        setattr(self._common_params, name, value)
        for p in self._params:
            setattr(p, name, value)

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
    def params(self, p: Union[PolymlpParamsSingle, list[PolymlpParamsSingle]]):
        """Setter of parameters."""
        if isinstance(p, PolymlpParamsSingle):
            self._params = [p]
        elif isinstance(p, (list, tuple, np.ndarray)):
            self._params = list(p)
        else:
            raise RuntimeError("Inappropriate input for parameters.")
        self._initialize()

    @property
    def n_type(self):
        """Return number of atom types."""
        return self._common_params.n_type

    @property
    def elements(self):
        """Return element strings."""
        return self._common_params.elements

    @property
    def atomic_energy(self):
        """Return atomic energies."""
        return self._common_params.atomic_energy

    @property
    def enable_spins(self):
        """Return whether spins are included or not."""
        return self._common_params.enable_spins

    @enable_spins.setter
    def enable_spins(self, include: bool):
        """Setter of enable_spins."""
        exception = ("phono3py", "sscha", "openmx", "electron")
        if self.dataset_type in exception and include:
            raise RuntimeError("Enable_spins not supported for given dataset type.")

        self.setter_common_property("enable_spins", include)

    @property
    def include_force(self):
        """Return whether forces are included or not."""
        return bool(self._common_params.include_force)

    @include_force.setter
    def include_force(self, include: bool):
        """Setter of include_force."""
        if self.dataset_type == "electron" and include:
            raise RuntimeError("Include_force not supported for given dataset type.")

        self.setter_common_property("include_force", include)

    @property
    def include_stress(self):
        """Return whether stresses are included or not."""
        return bool(self._common_params.include_stress)

    @include_stress.setter
    def include_stress(self, include: bool):
        """Setter of include_stress."""
        exception = ("phono3py", "sscha", "openmx", "electron")
        if self.dataset_type in exception and include:
            raise RuntimeError("Include_stress not supported for given dataset type.")

        self.setter_common_property("include_stress", include)

    @property
    def dataset_type(self):
        """Return dataset type."""
        return self._common_params.dataset_type

    @dataset_type.setter
    def dataset_type(self, dtype: str):
        """Setter of dataset type."""
        self.setter_common_property("dataset_type", dtype)
        self._set_tags()

    @property
    def temperature(self):
        """Return temperature."""
        return self._common_params.temperature

    @temperature.setter
    def temperature(self, temp: float):
        """Setter of temperature."""
        self.setter_common_property("temperature", temp)

    @property
    def electron_property(self):
        """Return target electronic property."""
        return self._common_params.electron_property

    @electron_property.setter
    def electron_property(self, prop: str):
        """Setter of target electronic property."""
        self.setter_common_property("electron_property", prop)

    @property
    def print_memory(self):
        """Return print_memory."""
        return bool(self._common_params.print_memory)

    @print_memory.setter
    def print_memory(self, pm: str):
        """Setter of print_memory."""
        self.setter_common_property("print_memory", pm)

    @property
    def solver_method(self):
        """Return solver method."""
        return self._common_params.solver_method

    @solver_method.setter
    def solver_method(self, method: str):
        """Setter of solver method."""
        self.setter_common_property("solver_method", method)

    @property
    def solver_alpha(self):
        """Return power indices of alpha penalty values."""
        return self._common_params.solver_alpha

    @solver_alpha.setter
    def solver_alpha(self, a: str):
        """Setter of print_memory."""
        self.setter_common_property("solver_alpha", a)

    @property
    def solver_alphas(self):
        """Return alpha penalty values."""
        return self._common_params.solver_alphas

    @solver_alphas.setter
    def solver_alphas(self, a: str):
        """Setter of print_memory."""
        self.setter_common_property("solver_alphas", a)

    @property
    def solver_use_gradient(self):
        """Return whether gradient solver is used."""
        return self._common_params.solver_use_gradient

    @solver_use_gradient.setter
    def solver_use_gradient(self, use_gradient: bool):
        """Setter of solver_use_gradient."""
        self.setter_common_property("solver_use_gradient", use_gradient)

    @property
    def solver_use_cv(self):
        """Return whether cross-validation is used in the solver."""
        return self._common_params.solver_use_cv

    @solver_use_cv.setter
    def solver_use_cv(self, use_cv: bool):
        """Setter of solver_use_cv."""
        self.setter_common_property("solver_use_cv", use_cv)

    @property
    def use_weight(self):
        """Return whether weight is used."""
        return self._common_params.use_weight

    @use_weight.setter
    def use_weight(self, use_weight: bool):
        """Setter of use_weight."""
        self.setter_common_property("use_weight", use_weight)

    @property
    def weight_stress(self):
        """Return weight stress."""
        return self._common_params.weight_stress

    @weight_stress.setter
    def weight_stress(self, weight_stress: float):
        """Setter of weight_stress."""
        self.setter_common_property("weight_stress", weight_stress)

    @property
    def weight_method(self):
        """Return weight method."""
        return self._common_params.weight_method

    @weight_method.setter
    def weight_method(self, weight_method: int):
        """Setter of weight_method."""
        self.setter_common_property("weight_method", weight_method)

    @property
    def weight_min_energy(self):
        """Return minimum energy for weight."""
        return self._common_params.weight_min_energy

    @weight_min_energy.setter
    def weight_min_energy(self, weight_min_energy: float):
        """Setter of weight_min_energy."""
        self.setter_common_property("weight_min_energy", weight_min_energy)

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

    def _set_tags(self):
        """Force to set tags."""
        if self._common_params is None:
            return self

        if self.dataset_type == "sscha":
            self.enable_spins = None
        elif self.dataset_type in ("phono3py", "openmx"):
            self.include_stress = False
            self.enable_spins = None
        elif self.dataset_type == "electron":
            self.include_force = False
            self.include_stress = False
            self.enable_spins = None
        return self
