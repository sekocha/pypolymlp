#!/usr/bin/env python

from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class PolymlpGtinvParams:
    """Dataclass of group-theoretical polynomial invariants."""

    order: int
    max_l: tuple[int]
    sym: tuple[bool]
    lm_seq: list
    l_comb: list
    lm_coeffs: list
    version: int = 1


@dataclass
class PolymlpModelParams:
    """Dataclass of input parameters for polymlp model.

    Parameters
    ----------
    cutoff: Cutoff radius.
    """

    cutoff: float
    model_type: int
    max_p: int
    max_l: int
    pair_params: list[list[float]]
    feature_type: Literal["pair", "gtinv"] = "gtinv"
    pair_type: str = "gaussian"
    gtinv_params: Optional[PolymlpGtinvParams] = None


@dataclass
class PolymlpParams:
    """Dataclass of input parameters for developing polymlp.

    Parameters
    ----------
    n_type: Number of atomic types.
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

    """
    Variables in params_dict
    ------------------------
      - n_type
      - include_force
      - include_stress
      - model
        - cutoff
        - model_type
        - max_p
        - max_l
        - feature_type
        - pair_type
        - pair_params
        - gtinv
          - order
          - max_l
          - lm_seq
          - l_comb
          - lm_coeffs
      - atomic_energy
      - reg
        - method
        - alpha
      - dft
        - train (vasprun locations)
        - test (vasprun locations)

    Variables in dft_dict (train_dft_dict, test_dft_dict)
    -----------------------------------------------------
        - energy
        - force
        - stress
        - structures
          - structure (1)
            - axis
            - positions
            - n_atoms
            - types
            - elements
          - ...
        - elements
        - volumes
        - total_n_atoms
    """
