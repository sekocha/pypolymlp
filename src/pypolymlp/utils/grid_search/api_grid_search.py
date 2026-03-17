"""API Class for using utility functions."""

import itertools
import os
from typing import Optional

import numpy as np

from pypolymlp.core.utils import get_atomic_size_scales
from pypolymlp.utils.grid_search.grid_enum import (
    add_complex_model1,
    enum_gtinv_models,
    enum_pair_models,
)
from pypolymlp.utils.grid_search.grid_io import save_params
from pypolymlp.utils.grid_search.grid_utils import GaussianAttrs, GtinvAttrs, ParamsGrid


class PolymlpGridSearch:
    """API Class for performing grid search for finding optimal MLPs.

    Examples
    --------
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
        self._elements = elements
        self._verbose = verbose

        self._grid = None
        self._grid_params = None
        self._grid_params_exp = None

    def _auto_cutoff(self):
        """Determine cutoff radii automatically."""
        sizes = get_atomic_size_scales()
        max_size = max([sizes[ele] for ele in self._elements])
        cutoffs = np.round(np.array([5.5, 7.5]) * max_size)
        cutoffs_ref = [6.0, 8.0]
        cutoffs = np.maximum(cutoffs, cutoffs_ref)
        return tuple(np.unique(cutoffs))

    def _auto_gaussians(self, cutoffs: tuple, nums_gaussians: Optional[tuple] = None):
        """Determine numbers of Gaussians automatically."""
        radial_params = []
        if nums_gaussians is None:
            for c, r in itertools.product(*[cutoffs, (1.0, 1.3)]):
                n1 = c + 1
                n_gauss = np.rint(n1 * r).astype(int)
                attr = GaussianAttrs(cutoff=c, n_gaussians=n_gauss)
                radial_params.append(attr)
        else:
            for c, n in itertools.product(*[cutoffs, nums_gaussians]):
                attr = GaussianAttrs(cutoff=c, n_gaussians=n)
                radial_params.append(attr)
        return radial_params

    def set_params(
        self,
        cutoffs: Optional[tuple] = None,
        nums_gaussians: Optional[tuple] = None,
        model_types: tuple = (3, 4),
        maxps: tuple = (2, 3),
        gtinv: bool = True,
        gtinv_order_ub: int = 4,
        gtinv_maxl_ub: tuple = (12, 8, 2, 1, 1),
        gtinv_maxl_int: tuple = (4, 4, 2, 1, 1),
        gtinv_attrs: Optional[list[GtinvAttrs]] = None,
        include_force: bool = True,
        include_stress: bool = True,
        regression_alpha: tuple = (-4, 1, 6),
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
        regression_alpha: Regularization parameters.
        """
        if cutoffs is None:
            cutoffs = self._auto_cutoff()
        radial_params = self._auto_gaussians(cutoffs, nums_gaussians)

        self._grid = ParamsGrid(
            radial_params=radial_params,
            model_types=model_types,
            maxps=maxps,
            gtinv=gtinv,
            gtinv_order_ub=gtinv_order_ub,
            gtinv_maxl_ub=gtinv_maxl_ub,
            gtinv_maxl_int=gtinv_maxl_int,
            gtinv_attrs=gtinv_attrs,
            include_force=include_force,
            include_stress=include_stress,
            regression_alpha=regression_alpha,
        )
        if self._verbose:
            print("Cutoff radius and number of Gaussians:", flush=True)
            for rad in radial_params:
                print("- cutoff:       ", rad.cutoff, flush=True)
                print("  n_gaussians:  ", rad.n_gaussians, flush=True)
            print("Polynomial orders:  ", maxps, flush=True)
            print("Invariant max order:", gtinv_order_ub, flush=True)
            print("Invariant max L:    ", gtinv_maxl_ub, flush=True)
        return self

    def run(self):
        """Enumerate models with pair and gtinv models."""
        self.enum_pair_models()
        if self._grid.gtinv:
            self.enum_gtinv_models()
        return self

    def enum_pair_models(self):
        """Enumerate models with pair features."""
        if self._grid is None:
            raise RuntimeError("Set parameter candidates at first. Use set_params.")

        if self._grid_params is None:
            self._grid_params = []

        params = enum_pair_models(self._grid, self._elements)
        self._grid_params.extend(params)
        return self

    def enum_gtinv_models(self):
        """Enumerate models with polynomial invariant features."""
        if self._grid is None:
            raise RuntimeError("Set parameter candidates at first. Use set_params.")

        if self._grid_params is None:
            self._grid_params = []

        params = enum_gtinv_models(self._grid, self._elements)
        self._grid_params.extend(params)
        return self

    def add_complex_models(self):
        """Enumerate complex models with many features."""
        if self._grid is None:
            raise RuntimeError("Set parameter candidates at first. Use set_params.")

        radial_params = self._grid.radial_params
        max_cutoff = max([rad.cutoff for rad in radial_params])
        max_n_gaussians = max([rad.n_gaussians for rad in radial_params])

        if max_n_gaussians < 15:
            n_gaussians = 15
            params = add_complex_model1(
                self._grid,
                self._elements,
                max_cutoff,
                n_gaussians,
            )
            self._grid_params.append(params)
        return self

    def save_models(self, path: str = "./polymlps", first_id: int = 1):
        """Save input files of models."""
        for i, params in enumerate(self._grid_params):
            path_mlp = path + "/polymlp-" + str(i + first_id).zfill(4)
            os.makedirs(path_mlp, exist_ok=True)
            save_params(params, path_mlp + "/polymlp.in")
        return self

    @property
    def grid(self):
        """Return parameters on grid."""
        return self._grid_params


# def save_hybrid_models(
#     grid1: list[PolymlpParams],
#     grid2: list[PolymlpParams],
#     path: str = "./polymlps_hybrid",
#     first_id: int = 1,
# ):
#     """Save input files of hybrid models."""
#     os.makedirs(path, exist_ok=True)
#     i = 0
#     for params1 in grid1:
#         for params2 in grid2:
#             path_mlp = path + "/polymlp-" + str(i + first_id).zfill(4)
#             os.makedirs(path_mlp, exist_ok=True)
#             save_params(params1, path_mlp + "/polymlp1.in")
#             save_params(params2, path_mlp + "/polymlp2.in")
#             i += 1
