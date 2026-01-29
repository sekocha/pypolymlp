"""Functions for computing X and y."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.dataset import DatasetList
from pypolymlp.mlp_dev.core.data_utils import PolymlpDataXY
from pypolymlp.mlp_dev.core.features import compute_features
from pypolymlp.mlp_dev.core.utils import get_min_energy
from pypolymlp.mlp_dev.core.utils_scales import round_scales
from pypolymlp.mlp_dev.core.utils_weights import apply_weights


def calc_xy(
    params: Union[PolymlpParams, list[PolymlpParams]],
    datasets: DatasetList,
    element_swap: bool = False,
    scales: Optional[np.ndarray] = None,
    min_energy: Optional[float] = None,
    weight_stress: float = 0.1,
    verbose: bool = False,
):
    """Calculate X and y data."""
    features = compute_features(
        params,
        datasets=datasets,
        element_swap=element_swap,
        verbose=verbose,
    )
    x = features.x
    first_indices = features.first_indices
    ne, nf, ns = features.n_data
    include_force = False if nf == 0 else True

    if verbose:
        print("Dataset size:", x.shape, flush=True)
        print("- n (energy) =", ne, flush=True)
        print("- n (force)  =", nf, flush=True)
        print("- n (stress) =", ns, flush=True)

    if scales is None:
        scales = np.std(x[:ne], axis=0)

    scales, zero_ids = round_scales(scales, include_force=include_force)
    x[:, zero_ids] = 0.0
    x /= scales

    if min_energy is None:
        min_energy = get_min_energy(datasets)

    y = np.zeros(x.shape[0])
    w = np.ones(x.shape[0])
    for data, indices in zip(datasets, first_indices):
        x, y, w = apply_weights(
            x,
            y,
            w,
            data,
            indices,
            weight_stress=weight_stress,
            min_e=min_energy,
        )

    data_xy = PolymlpDataXY(
        x=x,
        y=y,
        weights=w,
        scales=scales,
        min_energy=min_energy,
        first_indices=first_indices,
        cumulative_n_features=features.cumulative_n_features,
        n_structures=features.n_data[0],
    )
    return data_xy
