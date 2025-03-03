"""API Class for using utility functions."""

import itertools
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParams
from pypolymlp.utils.grid_search.grid_io import save_params


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
    max_l: int


@dataclass
class ParamsGrid:
    """Dataclass of parameters in grid search.

    Parameters
    ----------
    cutoff: Cutoff radius (Angstrom).
    nums_gaussians: Numbers of Gaussians.
    gtinv: Use settings for polynomial invariants.
    gtinv_order_ub: Upper bound of invariant order.
    gtinv_maxl_ub: Upper bound of invariant max_l.
    gtinv_maxl_int: Interval of invariant max_l
    gtinv_attrs: Possible candidates of invariant parameters.
    include_force: Include forces in regression.
    include_stress: Include stress in regression.
    reg_alpha_params: Regularization parameters.
    """

    cutoffs: tuple = (6.0, 8.0, 10.0)
    nums_gaussians: tuple = (7, 10, 13)
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
    reg_alpha_params: tuple = (-4, 3, 8)

    def __post_init__(self):
        """Post init method."""
        self._check_type()
        if self.gtinv:
            self._set_gtinv_params()
        else:
            self._set_pair_params()

    def _check_type(self):
        """Check types of variables."""
        if isinstance(self.cutoffs, float):
            self.cutoffs = [self.cutoffs]
        if isinstance(self.nums_gaussians, int):
            self.nums_gaussians = [self.nums_gaussians]
        if isinstance(self.model_types, int):
            self.model_types = [self.model_types]
        if isinstance(self.maxps, int):
            self.maxps = [self.maxps]
        if isinstance(self.gtinv_maxl_ub, int):
            self.gtinv_maxl_ub = [self.gtinv_maxl_ub]
        if isinstance(self.gtinv_maxl_int, int):
            self.gtinv_maxl_int = [self.gtinv_maxl_int]

        if len(self.reg_alpha_params) != 3:
            raise RuntimeError("len(reg_alpha_params) must be three.")

    def get_model_types_pair(self):
        """Set parameters for enumerating pair models"""
        return [t for t in self.model_types if t < 3]

    def _set_pair_params(self):
        """Set parameters for enumerating pair models"""
        self.model_types = [t for t in self.model_types if t < 3]

    def _set_gtinv_params(self):
        """Set gtinv_orders and gtinv_maxls"""
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


class PolymlpGridSearch:
    """API Class for performing grid search for finding optimal MLPs.

    Examples
    --------
    (Single models)
    grid1 = PolymlpGridSearch(elements=["Al"], verbose=True)
    grid1.set_params(
        cutoffs = (6.0, 8.0),
        nums_gaussians = (7, 10),
        model_types = (3, 4),
        gtinv = True,
        gtinv_order_ub = 3,
        gtinv_maxl_ub = (12, 8, 2, 1, 1),
        gtinv_maxl_int = (4, 4, 2, 1, 1),
        include_force = True,
        include_stress = True,
        reg_alpha_params = (-4, 3, 8),
    )
    grid1.enum_gtinv_models()
    grid1.save_models(path="./polymlps")

    (Hybrid models)
    grid2 = PolymlpGridSearch(elements=["Al"], verbose=True)
    grid2.set_params(
        cutoffs = (4.0),
        nums_gaussians = (4),
        model_types = (4),
        gaussian_width = 0.5,
        gtinv = True,
        gtinv_order_ub = 3,
        gtinv_maxl_ub = (12, 12, 2, 1, 1),
        gtinv_maxl_int = (6, 6, 2, 1, 1),
        include_force = True,
        include_stress = True,
        reg_alpha_params = (-4, 3, 8),
    )
    grid2.enum_gtinv_models()
    save_hybrid_models(grid1.grid, grid2.grid)
    """

    def __init__(self, elements: tuple, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._elements = elements
        self._grid = None
        self._grid_params = []

    def set_params(
        self,
        cutoffs: tuple = (6.0, 8.0, 10.0),
        nums_gaussians: tuple = (7, 10, 13),
        model_types: tuple = (2, 3, 4),
        maxps: tuple = (2, 3),
        gaussian_width: float = 1.0,
        gtinv: bool = True,
        gtinv_order_ub: int = 3,
        gtinv_maxl_ub: tuple = (12, 8, 2, 1, 1),
        gtinv_maxl_int: tuple = (4, 4, 2, 1, 1),
        gtinv_attrs: Optional[list[GtinvAttrs]] = None,
        include_force: bool = True,
        include_stress: bool = True,
        reg_alpha_params: tuple = (-4, 3, 8),
    ):
        """Initialize parameters in grid search for finding optimal MLPs.

        Parameters
        ----------
        cutoff: Cutoff radius (Angstrom).
        nums_gaussians: Numbers of Gaussians.
        gtinv: Use settings for polynomial invariants.
        gtinv_order_ub: Upper bound of invariant order.
        gtinv_maxl_ub: Upper bound of invariant max_l.
        gtinv_maxl_int: Interval of invariant max_l
        gtinv_attrs: Possible candidates of invariant parameters.
        include_force: Include forces in regression.
        include_stress: Include stress in regression.
        reg_alpha_params: Regularization parameters.
        """

        self._grid = ParamsGrid(
            cutoffs=cutoffs,
            nums_gaussians=nums_gaussians,
            model_types=model_types,
            maxps=maxps,
            gaussian_width=gaussian_width,
            gtinv=gtinv,
            gtinv_order_ub=gtinv_order_ub,
            gtinv_maxl_ub=gtinv_maxl_ub,
            gtinv_maxl_int=gtinv_maxl_int,
            include_force=include_force,
            include_stress=include_stress,
            reg_alpha_params=reg_alpha_params,
        )

    def enum_pair_models(self):
        """Enumerate models with pair features."""
        if self._grid is None:
            raise RuntimeError("No parameters found.")

        product = itertools.product(
            *[
                self._grid.cutoffs,
                self._grid.nums_gaussians,
                self._grid.get_model_types_pair(),
                self._grid.maxps,
            ]
        )
        for cut, n_gauss, model_type, mp in product:
            model = PolymlpModelParams(
                cutoff=cut,
                model_type=model_type,
                max_p=mp,
                max_l=0,
                feature_type="pair",
                pair_params_in1=(1.0, 1.0, 1),
                pair_params_in2=(0.0, cut - 1.0, n_gauss),
            )
            params = PolymlpParams(
                n_type=len(self._elements),
                elements=self._elements,
                model=model,
                regression_alpha=self._grid.reg_alpha_params,
                include_force=self._grid.include_force,
                include_stress=self._grid.include_stress,
            )
            self._grid_params.append(params)

        return self

    def enum_gtinv_models(self):
        """Enumerate models with polynomial invariant features."""
        if self._grid is None:
            raise RuntimeError("No parameters found.")

        product = itertools.product(
            *[
                self._grid.cutoffs,
                self._grid.nums_gaussians,
                self._grid.gtinv_attrs,
            ]
        )
        for cut, n_gauss, gtinv_attr in product:
            model = PolymlpModelParams(
                cutoff=cut,
                model_type=gtinv_attr.model_type,
                max_p=2,
                max_l=max(gtinv_attr.max_l),
                feature_type="gtinv",
                gtinv=gtinv_attr,
                pair_params_in1=(
                    self._grid.gaussian_width,
                    self._grid.gaussian_width,
                    1,
                ),
                pair_params_in2=(0, cut - 1.0, n_gauss),
            )
            params = PolymlpParams(
                n_type=len(self._elements),
                elements=self._elements,
                model=model,
                regression_alpha=self._grid.reg_alpha_params,
                include_force=self._grid.include_force,
                include_stress=self._grid.include_stress,
            )
            self._grid_params.append(params)

        return self

    def enum_complex_models(self):
        """Enumerate models with many features."""
        for cut in self._grid.cutoffs:
            gtinv_attr = GtinvAttrs(model_type=4, order=4, max_l=(12, 12, 4))
            model = PolymlpModelParams(
                cutoff=cut,
                model_type=4,
                max_p=2,
                max_l=12,
                feature_type="gtinv",
                gtinv=gtinv_attr,
                pair_params_in1=(
                    self._grid.gaussian_width,
                    self._grid.gaussian_width,
                    1,
                ),
                pair_params_in2=(0, cut - 1.0, 15),
            )
            params = PolymlpParams(
                n_type=len(self._elements),
                elements=self._elements,
                model=model,
                regression_alpha=self._grid.reg_alpha_params,
                include_force=self._grid.include_force,
                include_stress=self._grid.include_stress,
            )
            self._grid_params.append(params)

    def save_models(self, path: str = "./polymlps", first_id: int = 1):
        """Save input files of models."""
        os.makedirs(path, exist_ok=True)
        for i, params in enumerate(self._grid_params):
            path_mlp = path + "/polymlp-" + str(i + first_id).zfill(4)
            os.makedirs(path_mlp, exist_ok=True)
            save_params(params, path_mlp + "/polymlp.in")

        return self

    @property
    def grid(self):
        """Return parameters on grid."""
        return self._grid_params


def save_hybrid_models(
    grid1: list[PolymlpParams],
    grid2: list[PolymlpParams],
    path: str = "./polymlps_hybrid",
    first_id: int = 1,
):
    """Save input files of hybrid models."""
    os.makedirs(path, exist_ok=True)
    i = 0
    for params1 in grid1:
        for params2 in grid2:
            path_mlp = path + "/polymlp-" + str(i + first_id).zfill(4)
            os.makedirs(path_mlp, exist_ok=True)
            save_params(params1, path_mlp + "/polymlp1.in")
            save_params(params2, path_mlp + "/polymlp2.in")
            i += 1
