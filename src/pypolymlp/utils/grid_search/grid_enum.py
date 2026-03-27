"""Functions for enumerating models."""

import itertools

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParamsSingle
from pypolymlp.utils.grid_search.grid_utils import GtinvAttrs, ParamsGrid


def enum_pair_models(grid: ParamsGrid, elements: tuple):
    """Enumerate polynomial models with pair features."""
    model_types_pair = grid.get_model_types_pair()
    product = itertools.product(*[grid.radial_params, model_types_pair, grid.maxps])
    grid_params = []
    for rad, model_type, mp in product:
        model = PolymlpModelParams(
            cutoff=rad.cutoff,
            model_type=model_type,
            max_p=mp,
            max_l=0,
            feature_type="pair",
            n_gaussians=rad.n_gaussians,
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
    product = itertools.product(*[grid.radial_params, grid.gtinv_attrs])
    grid_params = []
    for rad, gtinv_attr in product:
        model = PolymlpModelParams(
            cutoff=rad.cutoff,
            model_type=gtinv_attr.model_type,
            max_p=2,
            max_l=max(gtinv_attr.max_l),
            feature_type="gtinv",
            gtinv=gtinv_attr,
            n_gaussians=rad.n_gaussians,
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


def add_single_model(
    grid: ParamsGrid,
    elements: tuple,
    cutoff: float = 8.0,
    n_gaussians: int = 10,
    max_p: int = 2,
    model_type: int = 4,
    gtinv_order: int = 3,
    gtinv_maxl: tuple = (8, 8),
):
    """Add complex model."""
    gtinv_attr = GtinvAttrs(model_type=model_type, order=gtinv_order, max_l=gtinv_maxl)
    model = PolymlpModelParams(
        cutoff=cutoff,
        model_type=model_type,
        max_p=max_p,
        max_l=max(gtinv_maxl),
        feature_type="gtinv",
        gtinv=gtinv_attr,
        n_gaussians=n_gaussians,
    )
    params = PolymlpParamsSingle(
        n_type=len(elements),
        elements=elements,
        model=model,
        regression_alpha=grid.regression_alpha,
        include_force=grid.include_force,
        include_stress=grid.include_stress,
    )
    return params
