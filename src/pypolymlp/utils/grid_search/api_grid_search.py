"""API Class for performing grid search for finding optimal MLPs."""

import itertools
import os
from typing import Optional

import numpy as np

from pypolymlp.core.utils import get_atomic_size_scales
from pypolymlp.utils.grid_search.grid_enum import (
    add_single_model,
    enum_gtinv_models,
    enum_pair_models,
)
from pypolymlp.utils.grid_search.grid_io import save_params
from pypolymlp.utils.grid_search.grid_utils import GaussianAttrs, GtinvAttrs, ParamsGrid


class PolymlpGridSearch:
    """API Class for performing grid search for finding optimal MLPs."""

    def __init__(self, elements: tuple, verbose: bool = False):
        """Init method."""
        self._elements = elements
        self._verbose = verbose

        self._grid = None
        self._grid_params = None
        self._grid_params_hybrid = None

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

        if max_n_gaussians >= 15:
            return self

        if max_n_gaussians < 12:
            n_gaussians = 12
        else:
            n_gaussians = 15

        params = add_single_model(
            self._grid,
            self._elements,
            cutoff=max_cutoff,
            n_gaussians=n_gaussians,
            gtinv_order=3,
            gtinv_maxl=(8, 8),
        )
        self._grid_params.append(params)

        params = add_single_model(
            self._grid,
            self._elements,
            cutoff=max_cutoff,
            n_gaussians=n_gaussians,
            gtinv_order=4,
            gtinv_maxl=(12, 8, 2),
        )
        self._grid_params.append(params)
        return self

    def enum_hybrid_models(self):
        """Enumerate hybrid models."""
        if self._grid is None:
            raise RuntimeError("Set parameter candidates at first. Use set_params.")
        if self._grid_params is None:
            raise RuntimeError("Enumerated models not found. Use enum_gtinv_models.")

        self._grid_params_hybrid = []
        for i, params_main in enumerate(self._grid_params):
            cutoff_main = params_main.model.cutoff
            cutoff = np.rint(cutoff_main * 2 / 3)
            n_gaussians = np.rint((cutoff + 1) * 1.3).astype(int)
            add_params = add_single_model(
                self._grid,
                self._elements,
                cutoff=cutoff,
                n_gaussians=n_gaussians,
                gtinv_order=3,
                gtinv_maxl=(4, 4),
            )
            self._grid_params_hybrid.append((params_main, add_params, i, 1))
            add_params = add_single_model(
                self._grid,
                self._elements,
                cutoff=cutoff,
                n_gaussians=n_gaussians,
                gtinv_order=3,
                gtinv_maxl=(8, 8),
            )
            self._grid_params_hybrid.append((params_main, add_params, i, 2))
        return self

    def save_models(self, path: str = "./polymlps", first_id: int = 1):
        """Save input files of models."""
        for i, params in enumerate(self._grid_params):
            path_mlp = path + "/polymlp-" + str(i + first_id).zfill(4)
            os.makedirs(path_mlp, exist_ok=True)
            save_params(params, path_mlp + "/polymlp.in")
        return self

    def save_hybrid_models(self, path: str = "./polymlps", first_id: int = 1):
        """Save input files of hybrid models."""
        for params1, params2, id_main, id_hyb in self._grid_params_hybrid:
            polyid = str(id_main + first_id).zfill(4)
            path_mlp = path + "/polymlp-" + polyid + "-h" + str(id_hyb).zfill(2)
            os.makedirs(path_mlp, exist_ok=True)
            save_params(params1, path_mlp + "/polymlp.in.1")
            save_params(params2, path_mlp + "/polymlp.in.2")
        return self

    @property
    def grid(self):
        """Return parameters on grid."""
        return self._grid_params

    @property
    def grid_hybrid(self):
        """Return parameters including hybrid models on grid."""
        return self._grid_params_hybrid
