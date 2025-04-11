"""Classes for computing X, y, X.T @ X and X.T @ y."""

from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpDataXY
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase
from pypolymlp.mlp_dev.core.utils_sequential import get_batch_slice
from pypolymlp.mlp_dev.core.utils_weights import apply_weights


def _round_scales(
    scales: np.ndarray, include_force: bool = True, threshold: float = 1e-10
):
    """Set scales so that they are not used for zero features."""
    if include_force:
        zero_ids = np.abs(scales) < threshold
    else:
        # Threshold value can be improved.
        zero_ids = np.abs(scales) < threshold * threshold
    scales[zero_ids] = 1.0
    return scales, zero_ids


class PolymlpDevDataXY(PolymlpDevDataXYBase):
    """Class for computing X and y."""

    def __init__(self, polymlp_dev_data: PolymlpDevData, verbose: bool = False):
        """Init method."""
        super().__init__(polymlp_dev_data, verbose=verbose)

    def run(self):
        """Compute X and y where weights and scales are applied."""
        self.compute_features()
        self.apply_scales()
        self.apply_weights()
        return self

    def compute_features(self):
        """Compute X for training and test datasets."""
        f_obj_train = self.features_class(
            self.params,
            self.train,
            print_memory=self.verbose,
        )
        f_obj_test = self.features_class(
            self.params,
            self.test,
            print_memory=self.verbose,
        )
        self.train_xy = f_obj_train.data_xy
        self.test_xy = f_obj_test.data_xy
        return self

    def print_data_shape(self):
        """Print dataset details."""
        if self._train_xy is None:
            raise ValueError("Not found: PolymlpDataXY.")

        x = self._train_xy.x
        ne, nf, ns = self._train_xy.n_data
        print("Training Dataset:", x.shape, flush=True)
        print("- n (energy) =", ne, flush=True)
        print("- n (force)  =", nf, flush=True)
        print("- n (stress) =", ns, flush=True)

        x = self._test_xy.x
        ne, nf, ns = self._test_xy.n_data
        print("Test Dataset:", x.shape, flush=True)
        print("- n (energy) =", ne, flush=True)
        print("- n (force)  =", nf, flush=True)
        print("- n (stress) =", ns, flush=True)
        return self

    def apply_scales(self):
        """Apply scales to X and y."""
        if self._train_xy is None:
            raise ValueError("Not found: PolymlpDataXY.")

        x = self._train_xy.x
        ne, nf, ns = self._train_xy.n_data
        self._scales = np.std(x[:ne], axis=0)

        self._scales, zero_ids = _round_scales(
            self._scales,
            include_force=self._common_params.include_force,
        )
        self._train_xy.x[:, zero_ids] = 0.0
        self._test_xy.x[:, zero_ids] = 0.0

        self._train_xy.x /= self._scales
        self._test_xy.x /= self._scales

        self._train_xy.scales = self._scales
        self._test_xy.scales = self._scales

        return self

    def apply_weights(self, weight_stress: float = 0.1):
        """Apply weights to X and y."""
        if self._train_xy is None:
            raise ValueError("Not found: PolymlpDataXY")

        self._train_xy = self._apply_weights_single_set(
            self._train,
            self._train_xy,
            weight_stress=weight_stress,
        )
        self._test_xy = self._apply_weights_single_set(
            self._test,
            self._test_xy,
            weight_stress=weight_stress,
        )
        return self

    def _apply_weights_single_set(
        self,
        data_dft: list[PolymlpDataDFT],
        data_xy: PolymlpDataXY,
        weight_stress: float = 0.1,
    ) -> PolymlpDataXY:
        """Apply weights to single set of (X, y)."""
        first_indices = data_xy.first_indices
        x = data_xy.x
        n_data, n_features = x.shape
        y = np.zeros(n_data)
        w = np.ones(n_data)

        for dft, indices in zip(data_dft, first_indices):
            x, y, w = apply_weights(
                x,
                y,
                w,
                dft,
                self._common_params,
                indices,
                weight_stress=weight_stress,
                min_e=self._min_energy,
            )

        data_xy.x = x
        data_xy.y = y
        data_xy.weight = w
        data_xy.scales = self._scales
        return data_xy


class PolymlpDevDataXYSequential(PolymlpDevDataXYBase):
    """Classes for computing X, y, X.T @ X and X.T @ y."""

    def __init__(self, polymlp_dev_data: PolymlpDevData, verbose: bool = False):
        """Init method."""
        super().__init__(polymlp_dev_data, verbose=verbose)
        self._n_features = None

    def run(
        self,
        batch_size: int = 128,
        n_features_threshold: int = 50000,
        element_swap: bool = False,
    ):
        """Compute X.T @ X and X.T @ y where weights and scales are applied."""
        self.run_train(
            batch_size=batch_size,
            n_features_threshold=n_features_threshold,
            element_swap=element_swap,
        )
        self.run_test(
            batch_size=batch_size,
            n_features_threshold=n_features_threshold,
            element_swap=element_swap,
        )
        return self

    def run_train(
        self,
        batch_size: int = 128,
        n_features_threshold: int = 50000,
        element_swap: bool = False,
    ):
        """Compute X.T @ X and X.T @ y for training."""
        self.train_xy = self.compute_products(
            self.train,
            scales=None,
            batch_size=batch_size,
            n_features_threshold=n_features_threshold,
            element_swap=element_swap,
        )
        return self

    def run_test(
        self,
        batch_size: int = 128,
        n_features_threshold: int = 50000,
        element_swap: bool = False,
    ):
        """Compute X.T @ X and X.T @ y for test."""
        self.test_xy = self.compute_products(
            self.test,
            scales=self._scales,
            batch_size=batch_size,
            n_features_threshold=n_features_threshold,
            element_swap=element_swap,
        )

    def compute_products(
        self,
        dft_list: list[PolymlpDataDFT],
        scales: Optional[np.ndarray] = None,
        batch_size: int = 128,
        n_features_threshold: int = 50000,
        element_swap: bool = False,
        n_batch: int = 10,
    ):
        """Compute X.T @ X and X.T @ y."""
        data_xy = PolymlpDataXY()
        for dft in dft_list:
            if self.verbose:
                print("----- Dataset:", dft.name, "-----", flush=True)
            n_str = len(dft.structures)
            dft = dft.sort()
            begin_ids, end_ids = get_batch_slice(n_str, batch_size)
            for begin, end in zip(begin_ids, end_ids):
                if self.verbose:
                    print("Structures:", end, "/", n_str, flush=True)
                data_xy = self._compute_products_single_batch(
                    dft.slice(begin, end),
                    data_xy,
                    scales=scales,
                    element_swap=element_swap,
                    n_features_threshold=n_features_threshold,
                    n_batch=n_batch,
                )

        if self._n_features > n_features_threshold:
            data_xy.xtx = self._symmetrize_xtx(data_xy.xtx, n_batch=n_batch)

        n_data = sum([len(d.energies) for d in dft_list])
        self._scales, zero_ids = self._compute_scales(
            scales, data_xy.xe_sum, data_xy.xe_sq_sum, n_data
        )
        data_xy.xtx[zero_ids] = 0.0
        data_xy.xtx[:, zero_ids] = 0.0
        data_xy.xty[zero_ids] = 0.0

        data_xy.xtx /= self._scales[:, np.newaxis]
        data_xy.xtx /= self._scales[np.newaxis, :]
        data_xy.xty /= self._scales
        data_xy.scales = self._scales

        return data_xy

    def _compute_scales(
        self,
        scales: np.array,
        xe_sum: np.ndarray,
        xe_sq_sum: np.ndarray,
        n_data: int,
    ):
        """Compute scales from xe_sum and xe_sq_sum."""
        if scales is None:
            variance = xe_sq_sum / n_data - np.square(xe_sum / n_data)
            variance[variance < 0.0] = 1.0
            self._scales = np.sqrt(variance)
        else:
            self._scales = scales

        self._scales, zero_ids = _round_scales(
            self._scales,
            include_force=self._common_params.include_force,
        )
        return self._scales, zero_ids

    def _compute_products_single_batch(
        self,
        dft_sliced: PolymlpDataDFT,
        data_xy: PolymlpDataXY,
        scales: Optional[np.ndarray] = None,
        element_swap: bool = False,
        n_features_threshold: int = 50000,
        n_batch: int = 10,
    ):
        """Compute X.T @ X and X.T @ y for a single batch."""
        features = self.features_class(
            self.params,
            dft_sliced,
            print_memory=self.verbose,
            element_swap=element_swap,
        )
        x = features.x
        first_indices = features.first_indices[0]
        ne, _, _ = features.n_data
        n_data, self._n_features = x.shape

        if self.verbose:
            peak = self._estimate_peak_memory(
                n_data, self._n_features, n_features_threshold
            )
            prefix = " Estimated peak memory allocation (X.T @ X, X):"
            print(prefix, np.round(peak * 8e-9, 2), "(GB)", flush=True)

        if scales is None:
            data_xy.xe_sum, data_xy.xe_sq_sum = self._compute_xe_sum(
                data_xy.xe_sum,
                data_xy.xe_sq_sum,
                x,
                ne,
            )

        y = np.zeros(n_data)
        w = np.ones(n_data)
        data_xy.total_n_data += n_data
        x, y, w = apply_weights(
            x,
            y,
            w,
            dft_sliced,
            self.common_params,
            first_indices,
            min_e=self.min_energy,
        )
        if self.verbose:
            print("Compute X.T @ X and X.T @ y", flush=True)
        if self._n_features < n_features_threshold:
            data_xy.xtx = self._sum_array(data_xy.xtx, x.T @ x)
        else:
            data_xy.xtx = self._sum_large_xtx(data_xy.xtx, x, n_batch=n_batch)

        data_xy.xty = self._sum_array(data_xy.xty, x.T @ y)
        data_xy.y_sq_norm += y @ y

        if self.is_hybrid:
            data_xy.cumulative_n_features = features.cumulative_n_features
        return data_xy

    def _estimate_peak_memory(
        self, n_data: int, n_features: int, n_features_threshold: int
    ):
        """Estimate peak memory required for allocating X and X.T @ X."""
        if n_features > n_features_threshold:
            peak_mem1 = (n_features**2) * 2
            peak_mem2 = n_features**2 + n_data * n_features
            peak_mem = max(peak_mem1, peak_mem2)
        else:
            peak_mem = (n_features**2) * 2 + n_data * n_features
        return peak_mem

    def _compute_xe_sum(
        self,
        xe_sum: np.ndarray,
        xe_sq_sum: np.ndarray,
        x: np.ndarray,
        ne: int,
    ):
        """Compute sums required for computing scales."""
        xe = x[:ne]
        xe_sum = self._sum_array(xe_sum, np.sum(xe, axis=0))
        xe_sq_sum = self._sum_array(xe_sq_sum, np.sum(np.square(xe), axis=0))
        return xe_sum, xe_sq_sum

    def _sum_array(self, array1: np.ndarray, array2: np.ndarray):
        """Add x.T @ x to xtx."""
        if array1 is None:
            return array2
        array1 += array2
        return array1

    def _sum_large_xtx(self, xtx: np.ndarray, x: np.ndarray, n_batch: int = 10):
        """Add x.T @ x to large xtx using batch calculations."""
        n_features = x.shape[1]
        if xtx is None:
            xtx = np.zeros((n_features, n_features))

        batch_size = max(100, n_features // n_batch)
        begin_ids, end_ids = get_batch_slice(n_features, batch_size)
        for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
            if self.verbose:
                print("Batch:", end_row, "/", n_features, flush=True)
            x1 = x[:, begin_row:end_row].T
            for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
                if i <= j:
                    prod = x1 @ x[:, begin_col:end_col]
                    xtx[begin_row:end_row, begin_col:end_col] += prod
        return xtx

    def _symmetrize_xtx(self, xtx: np.ndarray, n_batch: int = 10):
        """Symmetrize large matrix of xtx."""
        n_features = xtx.shape[0]
        batch_size = max(100, n_features // n_batch)
        begin_ids, end_ids = get_batch_slice(n_features, batch_size)
        for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
            for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
                if i > j:
                    xtx[begin_row:end_row, begin_col:end_col] = xtx[
                        begin_col:end_col, begin_row:end_row
                    ].T
        return xtx
