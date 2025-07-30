"""Classes for computing X, y, X.T @ X and X.T @ y."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.mlp_dev.core.features import compute_features
from pypolymlp.mlp_dev.core.features_attr import get_num_features
from pypolymlp.mlp_dev.core.utils import get_min_energy
from pypolymlp.mlp_dev.core.utils_scales import compute_scales, round_scales
from pypolymlp.mlp_dev.core.utils_sequential import (
    estimate_peak_memory,
    get_auto_batch_size,
    get_batch_slice,
    sum_array,
    sum_xtx,
    symmetrize_xtx,
)
from pypolymlp.mlp_dev.core.utils_weights import apply_weights


@dataclass
class PolymlpDataXY:
    """Dataclass of X, y, and related properties used for regression.

    Parameters
    ----------
    x: Predictor matrix, shape=(total_n_data, n_features)
    y: Observation vector, shape=(total_n_data)
    xtx: x.T @ x
    xty: x.T @ y
    scales: Scales of x, shape=(n_features)
    weights: Weights for data, shape=(total_n_data)
    n_data: Number of data (energy, force, stress)
    """

    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    xtx: Optional[np.ndarray] = None
    xty: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    min_energy: Optional[float] = None

    n_data: Optional[tuple[int, int, int]] = None
    first_indices: Optional[list[tuple[int, int, int]]] = None
    cumulative_n_features: Optional[int] = None

    xe_sum: Optional[np.ndarray] = None
    xe_sq_sum: Optional[np.ndarray] = None
    y_sq_norm: float = 0.0
    total_n_data: int = 0

    def clear_data(self):
        """Clear large data."""
        del self.x, self.y, self.xtx, self.xty
        del self.weights, self.xe_sum, self.xe_sq_sum

    def apply_scales(
        self,
        scales: Optional[np.ndarray] = None,
        include_force: bool = True,
    ):
        """Apply scales to X."""
        if self.x is None:
            raise RuntimeError("No data X found.")

        if scales is None:
            ne, nf, ns = self.n_data
            scales = np.std(self.x[:ne], axis=0)
        scales, zero_ids = round_scales(scales, include_force=include_force)

        self.x[:, zero_ids] = 0.0
        self.x /= scales
        self.scales = scales
        return self

    def apply_weights(
        self,
        common_params: PolymlpParams,
        dft_all: list[PolymlpDataDFT],
        min_energy: Optional[float] = None,
        weight_stress: float = 0.1,
    ):
        """Apply weights to X and y."""
        if self.x is None:
            raise RuntimeError("No data X found.")
        x = self.x
        n_data = x.shape[0]
        y = np.zeros(n_data)
        w = np.ones(n_data)

        if min_energy is None:
            min_energy = get_min_energy(dft_all)
        self.min_energy = min_energy

        for dft, indices in zip(dft_all, self.first_indices):
            x, y, w = apply_weights(
                x,
                y,
                w,
                dft,
                common_params,
                indices,
                weight_stress=weight_stress,
                min_e=min_energy,
            )

        self.x = x
        self.y = y
        self.weight = w
        return self

    def slices(self, n_samples: int, total_n_atoms: np.ndarray):
        """Return slices for selected data."""
        if self.x is None:
            raise RuntimeError("Data X is not found.")

        ids = list(range(n_samples))

        first_id = self.first_indices[0][2]
        ids_stress = range(first_id, first_id + n_samples * 6)
        ids.extend(ids_stress)

        first_id = self.first_indices[0][1]
        n_forces = sum(total_n_atoms[:n_samples]) * 3
        ids_force = range(first_id, first_id + n_forces)
        ids.extend(ids_force)
        ids = np.array(ids)
        return self.x[ids], self.y[ids]


def _get_features(
    params: Union[PolymlpParams, list[PolymlpParams]],
    dft: Optional[Union[PolymlpDataDFT, list[PolymlpDataDFT]]] = None,
    element_swap: bool = False,
    verbose: bool = True,
):
    """Calculate features and return features in DataXY format."""
    features = compute_features(
        params,
        dft,
        element_swap=element_swap,
        verbose=verbose,
    )
    xy = PolymlpDataXY(
        x=features.x,
        first_indices=features.first_indices,
        n_data=features.n_data,
        cumulative_n_features=features.cumulative_n_features,
    )
    return xy


def calc_xy(
    params: Union[PolymlpParams, list[PolymlpParams]],
    common_params: PolymlpParams,
    dft_all: list[PolymlpDataDFT],
    element_swap: bool = False,
    scales: Optional[np.ndarray] = None,
    min_energy: Optional[float] = None,
    weight_stress: float = 0.1,
    verbose: bool = False,
):
    """Calculate X and y data."""
    data_xy = _get_features(
        params,
        dft_all,
        element_swap=element_swap,
        verbose=verbose,
    )
    if verbose:
        ne, nf, ns = data_xy.n_data
        print("Dataset size:", data_xy.x.shape, flush=True)
        print("- n (energy) =", ne, flush=True)
        print("- n (force)  =", nf, flush=True)
        print("- n (stress) =", ns, flush=True)

    data_xy.apply_scales(scales=scales, include_force=common_params.include_force)
    data_xy.apply_weights(common_params, dft_all, min_energy=min_energy)
    return data_xy


def calc_xtx_xty(
    params: Union[PolymlpParams, list[PolymlpParams]],
    common_params: PolymlpParams,
    dft_all: list[PolymlpDataDFT],
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
        min_energy = get_min_energy(dft_all)

    data_xy = PolymlpDataXY()
    for dft in dft_all:
        if verbose:
            print("----- Dataset:", dft.name, "-----", flush=True)
        n_str = len(dft.structures)
        dft = dft.sort()
        begin_ids, end_ids = get_batch_slice(n_str, batch_size)
        for begin, end in zip(begin_ids, end_ids):
            if verbose:
                print("Structures:", end, "/", n_str, flush=True)
            data_xy = _compute_products_single_batch(
                params,
                common_params,
                dft.slice(begin, end),
                data_xy,
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

    n_data = sum([len(d.energies) for d in dft_all])
    scales, zero_ids = compute_scales(
        scales,
        data_xy.xe_sum,
        data_xy.xe_sq_sum,
        n_data,
        include_force=common_params.include_force,
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
    params: Union[PolymlpParams, list[PolymlpParams]],
    common_params: PolymlpParams,
    dft_sliced: PolymlpDataDFT,
    data_xy: PolymlpDataXY,
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
        dft_sliced,
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
        dft_sliced,
        common_params,
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
