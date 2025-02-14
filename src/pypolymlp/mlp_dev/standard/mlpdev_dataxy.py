"""Classes for computing X.T @ X and X.T @ y"""

from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpDataXY
from pypolymlp.mlp_dev.core.mlpdev_data import PolymlpDevData
from pypolymlp.mlp_dev.core.mlpdev_dataxy_base import PolymlpDevDataXYBase
from pypolymlp.mlp_dev.core.utils_sequential import get_batch_slice
from pypolymlp.mlp_dev.core.utils_weights import apply_weight_percentage


class PolymlpDevDataXY(PolymlpDevDataXYBase):

    def __init__(self, polymlp_dev_data: PolymlpDevData, verbose=True):
        super().__init__(polymlp_dev_data, verbose=verbose)

    def run(self):
        self.compute_features()
        self.apply_scales()
        self.apply_weights()
        return self

    def compute_features(self):
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


class PolymlpDevDataXYSequential(PolymlpDevDataXYBase):

    def __init__(self, polymlp_dev_data: PolymlpDevData, verbose=True):

        super().__init__(polymlp_dev_data, verbose=verbose)

        if not self.is_multiple_datasets:
            raise ValueError(
                "Sequential version is available "
                "for PolymlpDevParams with multiple datasets."
            )

        self._n_features = None

    def run(
        self,
        batch_size=128,
        n_features_threshold: int = 50000,
        element_swap=False,
    ):

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
        batch_size=128,
        n_features_threshold: int = 50000,
        element_swap=False,
    ):
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
        batch_size=128,
        n_features_threshold: int = 50000,
        element_swap=False,
    ):
        self.test_xy = self.compute_products(
            self.test,
            scales=self._scales,
            batch_size=batch_size,
            n_features_threshold=n_features_threshold,
            element_swap=element_swap,
        )

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

    def _compute_products_single_batch(
        self,
        dft_sliced: PolymlpDataDFT,
        data_xy: PolymlpDataXY,
        scales: Optional[np.ndarray] = None,
        element_swap: bool = False,
        n_features_threshold: int = 50000,
        n_batch: int = 10,
    ):
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
            print(
                " Estimated peak memory allocation (X.T @ X, X):",
                np.round(peak * 8e-9, 2),
                "(GB)",
                flush=True,
            )

        if scales is None:
            xe = x[:ne]
            data_xy.xe_sum = self._sum_array(data_xy.xe_sum, np.sum(xe, axis=0))
            data_xy.xe_sq_sum = self._sum_array(
                data_xy.xe_sq_sum, np.sum(np.square(xe), axis=0)
            )

        y = np.zeros(n_data)
        w = np.ones(n_data)
        data_xy.total_n_data += n_data

        x, y, w = apply_weight_percentage(
            x,
            y,
            w,
            dft_sliced,
            self.common_params,
            first_indices,
            min_e=self.min_energy,
        )
        if self.verbose:
            print("Compute X.T @ X", flush=True)
        if self._n_features > n_features_threshold:
            data_xy.xtx = self._sum_large_xtx(data_xy.xtx, x, n_batch=n_batch)
        else:
            data_xy.xtx = self._sum_array(data_xy.xtx, x.T @ x)

        if self.verbose:
            print("Compute X.T @ y", flush=True)
        data_xy.xty = self._sum_array(data_xy.xty, x.T @ y)
        data_xy.y_sq_norm += y @ y

        if self.is_hybrid:
            data_xy.cumulative_n_features = features.cumulative_n_features
        return data_xy

    def compute_products(
        self,
        dft_list: list[PolymlpDataDFT],
        scales: Optional[np.ndarray] = None,
        batch_size: int = 128,
        n_features_threshold: int = 50000,
        element_swap: bool = False,
        n_batch: int = 10,
    ):

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
                dft_sliced = dft.slice(begin, end)
                data_xy = self._compute_products_single_batch(
                    dft_sliced,
                    data_xy,
                    scales=scales,
                    element_swap=element_swap,
                    n_features_threshold=n_features_threshold,
                    n_batch=n_batch,
                )

        if self._n_features > n_features_threshold:
            data_xy.xtx = self._large_transpose_xtx(data_xy.xtx, n_batch=n_batch)

        if scales is None:
            n_data = sum([len(d.energies) for d in dft_list])
            variance = data_xy.xe_sq_sum / n_data - np.square(data_xy.xe_sum / n_data)
            self._scales = np.sqrt(variance)
        else:
            self._scales = scales

        self._scales[np.abs(self._scales) < 1e-20] = 1.0

        data_xy.xtx /= self._scales[:, np.newaxis]
        data_xy.xtx /= self._scales[np.newaxis, :]
        data_xy.xty /= self._scales
        data_xy.scales = self._scales

        return data_xy

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

        if n_features < n_batch:
            xtx += x.T @ x
            return xtx

        begin_ids, end_ids = get_batch_slice(n_features, n_features // n_batch)
        for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
            if self.verbose:
                print("Batch:", end_row, "/", n_features, flush=True)
            x1 = x[:, begin_row:end_row].T
            for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
                if i <= j:
                    prod = x1 @ x[:, begin_col:end_col]
                    xtx[begin_row:end_row, begin_col:end_col] += prod
        return xtx

    def _large_transpose_xtx(self, xtx: np.ndarray, n_batch: int = 10):
        n_features = xtx.shape[0]
        begin_ids, end_ids = get_batch_slice(n_features, n_features // n_batch)
        for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
            for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
                if i > j:
                    xtx[begin_row:end_row, begin_col:end_col] = xtx[
                        begin_col:end_col, begin_row:end_row
                    ].T
        return xtx
