"""API Class for using utility functions."""

import itertools
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class GtinvAttrs:
    """Dataclass of parameters for enumerating invariants.

    Parameters
    ----------
    model_type: Polynomial function type.
    order: Maximum order of polynomial invariants.
    max_l: Maximum angular numbers of polynomial invariants.
           [max_l for order=2, max_l for order=3, ...]
    """

    model_type: int
    order: int
    max_l: Union[list, tuple]


@dataclass
class GaussianAttrs:
    """Dataclass of parameters for setting Gaussians and cutoff radius.

    Parameters
    ----------
    cutoff: Cutoff radius.
    n_gaussians: Number of Gaussians.
    """

    cutoff: float
    n_gaussians: int


@dataclass
class ParamsGrid:
    """Dataclass of parameters in grid search.

    Parameters
    ----------
    radial_params: List of (cutoff radius in angstroms and number of Gaussians).
    gtinv: Use settings for polynomial invariants.
    gtinv_order_ub: Upper bound of invariant order.
    gtinv_maxl_ub: Upper bound of invariant max_l.
    gtinv_maxl_int: Interval of invariant max_l
    gtinv_attrs: Possible candidates of invariant parameters.
    include_force: Include forces in regression.
    include_stress: Include stress in regression.
    regression_alpha: Regularization parameters.
    """

    radial_params: list[GaussianAttrs]
    model_types: tuple = (2, 3, 4)
    maxps: tuple = (2, 3)
    gaussian_width: float = 1.0

    gtinv: bool = True
    gtinv_order_ub: int = 3
    gtinv_maxl_ub: tuple = (12, 8, 2, 1, 1)
    gtinv_maxl_int: tuple = (4, 4, 2, 1, 1)
    gtinv_attrs: Optional[list[GtinvAttrs]] = None

    include_force: bool = True
    include_stress: bool = True
    regression_alpha: tuple = (-4, 3, 8)

    def __post_init__(self):
        """Post init method."""
        self._check_type()
        if self.gtinv:
            self._set_gtinv_params()
        else:
            self._set_pair_params()

    def _check_type(self):
        """Check types of variables."""
        if isinstance(self.model_types, int):
            self.model_types = [self.model_types]
        if isinstance(self.maxps, int):
            self.maxps = [self.maxps]
        if isinstance(self.gtinv_maxl_ub, int):
            self.gtinv_maxl_ub = [self.gtinv_maxl_ub]
        if isinstance(self.gtinv_maxl_int, int):
            self.gtinv_maxl_int = [self.gtinv_maxl_int]

        if len(self.regression_alpha) != 3:
            raise RuntimeError("len(regression_alpha) must be three.")

    def get_model_types_pair(self):
        """Set parameters for enumerating pair models"""
        if self.gtinv:
            return [2]
        return [t for t in self.model_types if t < 3]

    def _set_pair_params(self):
        """Set parameters for enumerating pair models"""
        self.model_types = [t for t in self.model_types if t < 3]

    def _set_gtinv_params(self):
        """Set gtinv_orders and gtinv_maxls"""
        if self.gtinv_attrs is not None:
            return self.gtinv_attrs

        for l1, l2 in zip(self.gtinv_maxl_ub, self.gtinv_maxl_int):
            if l1 < l2:
                raise RuntimeError("maxl_ub must be larger than maxl_int.")

        maxl_list = []
        for gtinv_order in range(2, self.gtinv_order_ub + 1):
            idx = gtinv_order - 2
            interval, ub = self.gtinv_maxl_int[idx], self.gtinv_maxl_ub[idx]
            maxl_list.append(list(range(interval, ub + 1, interval)))

        l_list = []
        for gtinv_order in range(2, self.gtinv_order_ub + 1):
            end_idx = gtinv_order - 1
            l_prods = np.array(list(itertools.product(*maxl_list[:end_idx])))
            l_prods = [list(lp) for lp in l_prods if np.all(lp[:-1] - lp[1:] >= 0)]
            l_list.extend(l_prods)

        self.gtinv_attrs = []
        for model, lp in itertools.product(self.model_types, l_list):
            include = True
            if model == 2 and len(lp) > 2:
                include = False
            elif model == 2 and len(lp) == 1:
                include = False
            elif model == 2 and lp[1] > 5:
                include = False

            if include:
                attrs = GtinvAttrs(model_type=model, order=len(lp) + 1, max_l=lp)
                self.gtinv_attrs.append(attrs)
        return self.gtinv_attrs
