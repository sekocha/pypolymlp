"""API Class for using utility functions."""

import os
from typing import Optional

from pypolymlp.utils.grid_search.grid_enum import enum_gtinv_models, enum_pair_models
from pypolymlp.utils.grid_search.grid_io import save_params
from pypolymlp.utils.grid_search.grid_utils import GtinvAttrs, ParamsGrid


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
        regression_alpha: tuple = (-4, 3, 8),
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
        # TODO: Automatical cutoff distance determination using element size.

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
            gtinv_attrs=gtinv_attrs,
            include_force=include_force,
            include_stress=include_stress,
            regression_alpha=regression_alpha,
        )
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

    # def enum_complex_models(self):
    #     """Enumerate models with many features."""
    #     for cut in self._grid.cutoffs:
    #         gtinv_attr = GtinvAttrs(model_type=4, order=4, max_l=(12, 12, 4))
    #         model = PolymlpModelParams(
    #             cutoff=cut,
    #             model_type=4,
    #             max_p=2,
    #             max_l=12,
    #             feature_type="gtinv",
    #             gtinv=gtinv_attr,
    #             n_gaussians=15,
    #         )
    #         params = PolymlpParams(
    #             n_type=len(self._elements),
    #             elements=self._elements,
    #             model=model,
    #             regression_alpha=self._grid.regression_alpha,
    #             include_force=self._grid.include_force,
    #             include_stress=self._grid.include_stress,
    #         )
    #         self._grid_params.append(params)

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
