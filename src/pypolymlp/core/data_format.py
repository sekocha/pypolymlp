"""Dataclasses used for developing polymlp."""

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.utils import split_ids_train_test
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

    def __post_init__(self):
        self.check_errors()

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


@dataclass
class PolymlpDataDFT:
    """Dataclass of DFT dataset used for developing polymlp.

    Parameters
    ----------
    energies: Energies, shape=(n_str).
    forces: Forces, shape=(sum(n_atom(i_str) * 3)).
    stresses: Stress tensor elements, shape=(n_str * 6).
    volumes: Volumes, shape=(n_str).
    structures: Structures, list[PolymlpStructure]
    total_n_atoms: Numbers of atoms in structures.
    files: File names of structures.
    """

    energies: np.ndarray
    forces: np.ndarray
    stresses: np.ndarray
    volumes: np.ndarray
    structures: list[PolymlpStructure]
    total_n_atoms: np.ndarray
    files: list[str]
    elements: list[str]
    include_force: bool = True
    weight: float = 1.0
    name: str = "dataset"
    exist_force: bool = True
    exist_stress: bool = True

    def __post_init__(self):
        """Post init method."""
        self.check_errors()

    def check_errors(self):
        """Check errors."""
        assert self.energies.shape[0] * 6 == self.stresses.shape[0]
        assert self.energies.shape[0] == self.volumes.shape[0]
        assert self.energies.shape[0] == len(self.structures)
        assert self.energies.shape[0] == self.total_n_atoms.shape[0]
        assert self.energies.shape[0] == len(self.files)
        assert self.forces.shape[0] == np.sum(self.total_n_atoms) * 3

    def apply_atomic_energy(self, atom_e: tuple[float]):
        """Subtract atomic energies from energies."""
        atom_e = np.array(atom_e)
        self.energies = np.array(
            [e - st.n_atoms @ atom_e for e, st in zip(self.energies, self.structures)]
        )
        return self

    def slice(self, begin: int, end: int, name: str = "sliced"):
        """Slice DFT data in PolymlpDataDFT."""
        begin_f = sum(self.total_n_atoms[:begin]) * 3
        end_f = sum(self.total_n_atoms[:end]) * 3
        dft_dict_sliced = PolymlpDataDFT(
            energies=self.energies[begin:end],
            forces=self.forces[begin_f:end_f],
            stresses=self.stresses[begin * 6 : end * 6],
            volumes=self.volumes[begin:end],
            structures=self.structures[begin:end],
            total_n_atoms=self.total_n_atoms[begin:end],
            files=self.files[begin:end],
            elements=self.elements,
            include_force=self.include_force,
            weight=self.weight,
            name=name,
        )
        return dft_dict_sliced

    def _force_stress_ids(self, ids: np.ndarray):
        """Return IDs for force and stress corresponding to IDs for energy."""
        force_end = np.cumsum(self.total_n_atoms * 3)
        force_begin = np.insert(force_end[:-1], 0, 0)
        ids_force = np.array(
            [i for b, e in zip(force_begin[ids], force_end[ids]) for i in range(b, e)]
        )
        ids_stress = ((ids * 6)[:, None] + np.arange(6)[None, :]).reshape(-1)
        return ids_force, ids_stress

    def sort(self):
        """Sort DFT data in terms of the number of atoms."""
        ids = np.argsort(self.total_n_atoms)
        ids_force, ids_stress = self._force_stress_ids(ids)

        self.energies = self.energies[ids]
        self.forces = self.forces[ids_force]
        self.stresses = self.stresses[ids_stress]
        self.volumes = self.volumes[ids]
        self.total_n_atoms = self.total_n_atoms[ids]
        self.structures = [self.structures[i] for i in ids]
        self.files = [self.files[i] for i in ids]
        return self

    def split(self, train_ratio: float = 0.9):
        """Split dataset into training and test datasets."""
        train_ids, test_ids = split_ids_train_test(len(self.energies))
        ids_force, ids_stress = self._force_stress_ids(train_ids)
        train = PolymlpDataDFT(
            energies=self.energies[train_ids],
            forces=self.forces[ids_force],
            stresses=self.stresses[ids_stress],
            volumes=self.volumes[train_ids],
            structures=[self.structures[i] for i in train_ids],
            total_n_atoms=self.total_n_atoms[train_ids],
            files=[self.files[i] for i in train_ids],
            elements=self.elements,
            include_force=self.include_force,
            weight=self.weight,
            name=self.name,
        )
        ids_force, ids_stress = self._force_stress_ids(test_ids)
        test = PolymlpDataDFT(
            energies=self.energies[test_ids],
            forces=self.forces[ids_force],
            stresses=self.stresses[ids_stress],
            volumes=self.volumes[test_ids],
            structures=[self.structures[i] for i in test_ids],
            total_n_atoms=self.total_n_atoms[test_ids],
            files=[self.files[i] for i in test_ids],
            elements=self.elements,
            include_force=self.include_force,
            weight=self.weight,
            name=self.name,
        )
        return train, test


@dataclass
class PolymlpDataXY:
    """Dataclass of X, y, and related properties used for regression.

    Parameters
    ----------
    x: Predictor matrix, shape=(total_n_data, n_features)
    y: Observation vector, shape=(total_n_data)
    xtx: x.T @ x
    xty: x.T @ y
    scales: Scales of x, shape=(n_features)
    weights: Weights for data, shape=(total_n_data)
    n_data: Number of data (energy, force, stress)
    """

    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    xtx: Optional[np.ndarray] = None
    xty: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    n_data: Optional[tuple[int, int, int]] = None
    first_indices: Optional[list[tuple[int, int, int]]] = None
    cumulative_n_features: Optional[int] = None
    xe_sum: Optional[np.ndarray] = None
    xe_sq_sum: Optional[np.ndarray] = None
    y_sq_norm: float = 0.0
    total_n_data: int = 0


@dataclass
class PolymlpDataMLP:
    """Dataclass of regression results.

    Parameters
    ----------
    coeffs: MLP coefficients, shape=(n_features).
    scales: Scales of x, shape=(n_features).
    """

    coeffs: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    rmse: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    predictions_train: Optional[np.ndarray] = None
    predictions_test: Optional[np.ndarray] = None
    error_train: Optional[dict] = None
    error_test: Optional[dict] = None
