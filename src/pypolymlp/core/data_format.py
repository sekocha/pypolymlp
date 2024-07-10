"""Dataclasses used for developing polymlp."""

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Self, Union

import numpy as np


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
    volume: float
    supercell_matrix: Optional[np.ndarray] = None
    positions_cartesian: Optional[np.ndarray] = None
    valence: Optional[list] = None
    n_unitcells: Optional[int] = None
    comment: Optional[str] = None


@dataclass
class PolymlpGtinvParams:
    """Dataclass of group-theoretical polynomial invariants.

    Parameters
    ----------
    order: Maximum order of polynomial invariants.
    max_l: Maximum angular numbers of polynomial invariants.
           [maxl for order=2, maxl for order=3, ...]
    """

    order: int
    max_l: tuple[int]
    sym: tuple[bool]
    lm_seq: list
    l_comb: list
    lm_coeffs: list
    version: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


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
    model_type: int
    max_p: int
    max_l: int
    pair_params: list[list[float]]
    feature_type: Literal["pair", "gtinv"] = "gtinv"
    pair_type: str = "gaussian"
    gtinv: Optional[PolymlpGtinvParams] = None

    def as_dict(self) -> dict:
        model_dict = asdict(self)
        if self.gtinv is not None:
            model_dict["gtinv"] = self.gtinv.as_dict()
        return model_dict


@dataclass
class PolymlpParams:
    """Dataclass of input parameters for developing polymlp.

    Parameters
    ----------
    n_type: Number of atomic types.
    elements: Element species, (e.g., ['Mg','O']).
    atomic_energy: Atomic energies (in eV).
    include_force: Considering force entries.
    include_stress: Considering stress entries.
    regression_alpha: Parameters for penalty term in linear ridge regression.
                      Parameters are given as np.linspace(p[0], p[1], p[2]).
    """

    n_type: int
    elements: tuple[str]
    atomic_energy: tuple[float]
    model: PolymlpModelParams
    dft_train: Optional[Union[list, dict]] = None
    dft_test: Optional[Union[list, dict]] = None
    regression_method: str = "ridge"
    regression_alpha: tuple[float, float, int] = (-3.0, 1.0, 5)
    include_force: bool = True
    include_stress: bool = True
    dataset_type: Literal["vasp", "phonon3py"] = "vasp"
    element_order: Optional[tuple[str]] = None
    element_swap: bool = False

    def as_dict(self) -> dict:
        params_dict = asdict(self)
        params_dict["model"] = self.model.as_dict()
        return params_dict


@dataclass
class PolymlpDFTDataset:
    """Dataclass of DFT dataset used for developing polymlp.

    Parameters
    ----------
    energies: Energies, shape=(n_str).
    forces: Forces, shape=(sum(n_atom(i_str) * 3)).
    stresses: Stress tensor elements, shape=(n_str * 6).
    volumes: Volumes, shape=(n_str).
    structures: Structures, list[PolymlpStructure]
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

    def apply_atomic_energy(self, atom_e: tuple[float]) -> Self:
        atom_e = np.array(atom_e)
        self.energies = np.array(
            [e - st.n_atoms @ atom_e for e, st in zip(self.energies, self.structures)]
        )
        return self
