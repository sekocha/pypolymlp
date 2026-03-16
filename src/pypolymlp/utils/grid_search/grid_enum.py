"""Functions for enumerating models."""

import itertools

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParamsSingle
from pypolymlp.utils.grid_search.grid_utils import ParamsGrid


def enum_pair_models(grid: ParamsGrid, elements: tuple):
    """Enumerate polynomial models with pair features."""
    model_types_pair = grid.get_model_types_pair()
    product = itertools.product(
        *[grid.cutoffs, grid.nums_gaussians, model_types_pair, grid.maxps]
    )
    grid_params = []
    for cut, n_gauss, model_type, mp in product:
        model = PolymlpModelParams(
            cutoff=cut,
            model_type=model_type,
            max_p=mp,
            max_l=0,
            feature_type="pair",
            n_gaussians=n_gauss,
        )
        params = PolymlpParamsSingle(
            n_type=len(elements),
            elements=elements,
            model=model,
            regression_alpha=grid.regression_alpha,
            include_force=grid.include_force,
            include_stress=grid.include_stress,
        )
        grid_params.append(params)

    return grid_params


def enum_gtinv_models(grid: ParamsGrid, elements: tuple):
    """Enumerate polynomial models with invariant features."""
    product = itertools.product(*[grid.cutoffs, grid.nums_gaussians, grid.gtinv_attrs])
    grid_params = []
    for cut, n_gauss, gtinv_attr in product:
        model = PolymlpModelParams(
            cutoff=cut,
            model_type=gtinv_attr.model_type,
            max_p=2,
            max_l=max(gtinv_attr.max_l),
            feature_type="gtinv",
            gtinv=gtinv_attr,
            n_gaussians=n_gauss,
        )
        params = PolymlpParamsSingle(
            n_type=len(elements),
            elements=elements,
            model=model,
            regression_alpha=grid.regression_alpha,
            include_force=grid.include_force,
            include_stress=grid.include_stress,
        )
        grid_params.append(params)

    return grid_params
