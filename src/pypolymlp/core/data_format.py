"""Dataclasses used for developing polymlp."""

import copy
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.cxx.lib import libmlpcpp


@dataclass
class PolymlpStructure:
    """Dataclass of structure.

    Parameters
    ----------
    axis: Axis matrix [a, b, c], shape=(3, 3).
    positions: Scaled positions, shape=(3, n_atom).
    n_atoms: Number of atoms, (e.g.) [4, 4]
    elements: Element list, (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
    types: Atomic type integers, (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
    volume: Cell volume in ang.^3
    """

    axis: np.ndarray
    positions: np.ndarray
    n_atoms: Union[np.ndarray, list]
    elements: Union[np.ndarray, list]
    types: Union[np.ndarray, list]
    volume: Optional[float] = None
    supercell_matrix: Optional[np.ndarray] = None
    positions_cartesian: Optional[np.ndarray] = None
    valence: Optional[list] = None
    n_unitcells: Optional[int] = None
    axis_inv: Optional[np.ndarray] = None
    comment: Optional[str] = None
    name: Optional[str] = None

    masses: Optional[float] = None
    velocities: Optional[np.ndarray] = None
    momenta: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.volume is None:
            self.volume = np.linalg.det(self.axis)
        self.check_errors()

    def check_errors(self):
        """Check errors."""
        self.axis = np.array(self.axis)
        self.positions = np.array(self.positions)
        assert self.axis.shape[0] == 3
        assert self.axis.shape[1] == 3
        assert self.positions.shape[0] == 3
        assert self.positions.shape[1] == len(self.elements)
        assert len(self.elements) == len(self.types)
        assert len(self.elements) == sum(self.n_atoms)

    def set_positions_cartesian(self):
        """Calculate positions_cartesian."""
        self.positions_cartesian = self.axis @ self.positions

    def reorder(self):
        """Reorder positions, types, and elements according to types."""
        map_elements = dict()
        for t, e in zip(self.types, self.elements):
            map_elements[t] = e

        n_atoms, positions_reorder, types_reorder = [], [], []
        for i in sorted(set(self.types)):
            ids = np.array(self.types) == i
            n_atoms.append(np.count_nonzero(ids))
            positions_reorder.extend(self.positions.T[ids])
            types_reorder.extend(np.array(self.types)[ids])

        st = copy.deepcopy(self)
        st.positions = np.array(positions_reorder).T
        st.n_atoms = n_atoms
        st.types = types_reorder
        st.elements = [map_elements[t] for t in types_reorder]
        return st


@dataclass
class PolymlpGtinvParams:
    """Dataclass of group-theoretical polynomial invariants.

    Parameters
    ----------
    order: Maximum order of polynomial invariants.
    max_l: Maximum angular numbers of polynomial invariants.
           [maxl for order=2, maxl for order=3, ...]
    n_type: Number of atom types.

    lm_seq, l_comb, and lm_coeffs are automatically generated for given max_l.
    """

    order: int
    max_l: tuple[int]
    n_type: int
    sym: tuple[bool] = (False, False, False, False, False)
    lm_seq: Optional[list] = None
    l_comb: Optional[list] = None
    lm_coeffs: Optional[list] = None
    version: int = 1

    def __post_init__(self):
        self.check_errors()
        if self.order > 0:
            self.get_invariants()
        else:
            self.lm_seq, self.l_comb, self.lm_coeffs = [], [], []

    def as_dict(self) -> dict:
        """Convert the dataclass to dict."""
        return asdict(self)

    def check_errors(self):
        """Check errors."""
        size = self.order - 1
        assert len(self.max_l) >= size

    def get_invariants(self):
        """Read polynomial invariants."""
        rgi = libmlpcpp.Readgtinv(
            self.order,
            self.max_l,
            self.sym,
            self.n_type,
            self.version,
        )
        self.lm_seq = rgi.get_lm_seq()
        self.l_comb = rgi.get_l_comb()
        self.lm_coeffs = rgi.get_lm_coeffs()


@dataclass
class PolymlpModelParams:
    """Dataclass of input parameters for polymlp model.

    Parameters
    ----------
    cutoff: Cutoff radius (Angstrom).
    model_type: Polynomial function type.
        model_type = 1: Linear polynomial of polynomial invariants
        model_type = 2: Polynomial of polynomial invariants
        model_type = 3: Polynomial of pair invariants
                        + linear polynomial of polynomial invariants
        model_type = 4: Polynomial of pair and second-order invariants
                        + linear polynomial of polynomial invariants
    max_p: Order of polynomial function.
    max_l: Maximum angular number.
    feature_type: Feature type, 'gtinv' or 'pair'.
    gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
        Parameters are given as np.linspace(p[0], p[1], p[2]),
        where p[0], p[1], and p[2] are given by gaussian_params1
        and gaussian_params2.
    """

    cutoff: float
    model_type: Literal[1, 2, 3, 4]
    max_p: Literal[1, 2, 3]
    max_l: int
    feature_type: Literal["pair", "gtinv"] = "gtinv"
    gtinv: Optional[PolymlpGtinvParams] = None
    pair_type: str = "gaussian"
    pair_conditional: bool = False
    pair_params: Optional[list[list[float]]] = None
    pair_params_conditional: Optional[dict] = None
    pair_params_in1: Optional[tuple] = None
    pair_params_in2: Optional[tuple] = None

    def __post_init__(self):
        self.check_errors()

    def as_dict(self) -> dict:
        """Convert the dataclass to dict."""
        model_dict = asdict(self)
        if self.gtinv is not None:
            model_dict["gtinv"] = self.gtinv.as_dict()
        return model_dict

    def check_errors(self):
        if self.pair_params_in1 is None:
            if self.pair_params is None and self.pair_params_conditional is None:
                raise KeyError(
                    "Either of pair_params or pair_params_conditional is required."
                )


@dataclass
class PolymlpParams:
    """Dataclass of input parameters for developing polymlp.

    Parameters
    ----------
    n_type: Number of atomic types.
    elements: Element species, (e.g., ['Mg','O']).
    model: Model parameters in PolymlpModelParams.
    atomic_energy: Atomic energies (in eV).
    dft_train, dft_test: Location of DFT datasets.
                         Their data structures depend on the dataset type.
    regression_alpha: Parameters for penalty term in linear ridge regression.
                      alphas = np.linspace(p[0], p[1], p[2]).
    include_force: Consider force entries.
    include_stress: Consider stress entries.
    temperature: Temperature (active if dataset = "electron")
    electron_property: Target electronic property
    """

    n_type: int
    elements: tuple[str]
    model: PolymlpModelParams
    atomic_energy: Optional[tuple[float]] = None
    dft_train: Optional[Union[list, dict]] = None
    dft_test: Optional[Union[list, dict]] = None
    regression_method: str = "ridge"
    regression_alpha: tuple[float] = tuple(np.linspace(-3.0, 1.0, 5))
    include_force: bool = True
    include_stress: bool = True
    dataset_type: Literal["vasp", "phonon3py"] = "vasp"
    element_order: Optional[tuple[str]] = None
    element_swap: bool = False
    print_memory: bool = False
    type_indices: Optional[list] = None
    type_full: Optional[bool] = None
    temperature: float = 300
    electron_property: Literal[
        "free_energy",
        "energy",
        "entropy",
        "specific_heat",
    ] = "free_energy"
    name: Optional[str] = None
    mass: Optional[float] = None
    priority_infile: Optional[str] = None
    alphas: Optional[np.ndarray] = None

    def __post_init__(self):
        self.check_errors()
        self.alphas = np.array([pow(10, a) for a in self.regression_alpha])

    def as_dict(self) -> dict:
        """Convert the dataclass to dict."""
        params_dict = asdict(self)
        params_dict["model"] = self.model.as_dict()
        return params_dict

    def check_errors(self):
        """Check errors."""
        assert len(self.elements) == self.n_type
        if self.atomic_energy is not None:
            assert len(self.atomic_energy) == self.n_type

    def set_alphas(self, reg_alpha_params: tuple):
        """Set alpha values."""
        self.regression_alpha = np.linspace(
            reg_alpha_params[0],
            reg_alpha_params[1],
            reg_alpha_params[2],
        )
        self.alphas = np.array([pow(10, a) for a in self.regression_alpha])
