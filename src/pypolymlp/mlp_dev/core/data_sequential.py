"""Functions for computing X.T @ X and X.T @ y sequentially."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.dataset import Dataset, DatasetList
from pypolymlp.mlp_dev.core.data_utils import PolymlpDataXY
from pypolymlp.mlp_dev.core.features import compute_features
from pypolymlp.mlp_dev.core.features_attr import get_num_features
from pypolymlp.mlp_dev.core.utils import get_min_energy
from pypolymlp.mlp_dev.core.utils_scales import compute_scales
from pypolymlp.mlp_dev.core.utils_sequential import (
    estimate_peak_memory,
    get_auto_batch_size,
    get_batch_slice,
    sum_array,
    sum_xtx,
    symmetrize_xtx,
)
from pypolymlp.mlp_dev.core.utils_weights import apply_weights


def calc_xtx_xty(
    params: Union[PolymlpParams, list[PolymlpParams]],
    datasets: DatasetList,
    element_swap: bool = False,
    scales: Optional[np.ndarray] = None,
    min_energy: Optional[float] = None,
    weight_stress: float = 0.1,
    batch_size: Optional[int] = None,
    use_gradient: bool = False,
    n_features_threshold: int = 50000,
    verbose: bool = False,
):
    """Compute X.T @ X and X.T @ y."""
    n_features = get_num_features(params)
    if batch_size is None:
        batch_size = get_auto_batch_size(
            n_features,
            use_gradient=use_gradient,
            verbose=verbose,
        )
    if min_energy is None:
        min_energy = get_min_energy(datasets)

    data_xy = PolymlpDataXY()
    for data in datasets:
        if verbose:
            print("----- Dataset:", data.name, "-----", flush=True)
        data.sort_dft()
        n_str = len(data.structures)
        begin_ids, end_ids = get_batch_slice(n_str, batch_size)
        for begin, end in zip(begin_ids, end_ids):
            if verbose:
                print("Structures:", end, "/", n_str, flush=True)

            sliced_data = data.slice_dft(begin, end)
            data_xy = _compute_products_single_batch(
                data_xy,
                params,
                sliced_data,
                element_swap=element_swap,
                scales=scales,
                min_energy=min_energy,
                weight_stress=weight_stress,
                use_gradient=use_gradient,
                n_features_threshold=n_features_threshold,
                verbose=verbose,
            )

    if n_features > n_features_threshold:
        data_xy.xtx = symmetrize_xtx(data_xy.xtx)

    n_data = sum([len(d.energies) for d in datasets])
    scales, zero_ids = compute_scales(
        scales,
        data_xy.xe_sum,
        data_xy.xe_sq_sum,
        n_data,
        include_force=datasets.include_force,
    )
    data_xy.xtx[zero_ids] = 0.0
    data_xy.xtx[:, zero_ids] = 0.0
    data_xy.xty[zero_ids] = 0.0

    data_xy.xtx /= scales[:, np.newaxis]
    data_xy.xtx /= scales[np.newaxis, :]
    data_xy.xty /= scales
    data_xy.scales = scales
    data_xy.min_energy = min_energy

    return data_xy


def _compute_products_single_batch(
    data_xy: PolymlpDataXY,
    params: Union[PolymlpParams, list[PolymlpParams]],
    dataset_sliced: Dataset,
    element_swap: bool = False,
    scales: Optional[np.ndarray] = None,
    min_energy: Optional[float] = None,
    weight_stress: float = 0.1,
    use_gradient: bool = False,
    n_features_threshold: int = 50000,
    verbose: bool = False,
):
    """Compute X.T @ X and X.T @ y for a single batch."""
    features = compute_features(
        params,
        structures=dataset_sliced.structures,
        element_swap=element_swap,
        verbose=verbose,
    )
    x = features.x
    first_indices = features.first_indices[0]
    n_data, n_features = x.shape

    if verbose:
        peak = estimate_peak_memory(
            n_data,
            n_features,
            n_features_threshold,
            use_gradient=use_gradient,
        )
        prefix = " Estimated peak memory allocation (X.T @ X, X):"
        print(prefix, np.round(peak, 2), "(GB)", flush=True)

    if scales is None:
        ne, _, _ = features.n_data
        xe = x[:ne]
        data_xy.xe_sum = sum_array(data_xy.xe_sum, np.sum(xe, axis=0))
        data_xy.xe_sq_sum = sum_array(data_xy.xe_sq_sum, np.sum(np.square(xe), axis=0))

    y = np.zeros(n_data)
    w = np.ones(n_data)
    x, y, w = apply_weights(
        x,
        y,
        w,
        dataset_sliced,
        first_indices,
        weight_stress=weight_stress,
        min_e=min_energy,
    )
    data_xy.xtx = sum_xtx(
        data_xy.xtx,
        x,
        n_features_threshold=n_features_threshold,
        verbose=verbose,
    )
    data_xy.xty = sum_array(data_xy.xty, x.T @ y)
    data_xy.y_sq_norm += y @ y
    data_xy.total_n_data += n_data
    data_xy.cumulative_n_features = features.cumulative_n_features

    return data_xy
