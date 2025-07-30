"""Utility functions for sequential regression."""

import os

import numpy as np


def estimate_peak_memory(
    n_data: int,
    n_features: int,
    n_features_threshold: int,
    use_gradient: bool = False,
):
    """Estimate peak memory required for allocating X and X.T @ X."""
    if use_gradient:
        peak_mem = n_features**2 + n_data * n_features
    else:
        peak_mem1 = (n_features**2) * 2
        if n_features > n_features_threshold:
            peak_mem2 = n_features**2 + n_data * n_features
            peak_mem = max(peak_mem1, peak_mem2)
        else:
            peak_mem = peak_mem1 + n_data * n_features
    return peak_mem * 8e-9


def get_batch_slice(n_data: int, batch_size: int):
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch


def get_auto_batch_size(
    n_features: int,
    n_features_threshold: int = 50000,
    use_gradient: bool = False,
    verbose: bool = False,
):
    """Return optimal batch size determined automatically.

    Allocate roughly double amount of memory for X.T @ X
    and memory for X of size (n_str * N3, n_features).
    """
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    if use_gradient:
        mem_bytes = mem_bytes * 0.8 - (n_features**2 * 8)
    else:
        mem_bytes = mem_bytes * 0.8 - (n_features**2 * 8) * 2

    if mem_bytes < 0:
        raise RuntimeError("Large amount of memory is required for computing X.")

    N3, max_mem = 450, 4e11
    batch_size = round(min(mem_bytes, max_mem) / (N3 * n_features * 8))
    return batch_size


def sum_array(array1: np.ndarray, array2: np.ndarray):
    """Add x.T @ x to xtx."""
    if array1 is None:
        return array2
    array1 += array2
    return array1


def sum_large_xtx(
    xtx: np.ndarray,
    x: np.ndarray,
    n_batch: int = 10,
    verbose: bool = False,
):
    """Add x.T @ x to large xtx using batch calculations."""
    n_features = x.shape[1]
    if xtx is None:
        xtx = np.zeros((n_features, n_features))

    batch_size = max(100, n_features // n_batch)
    begin_ids, end_ids = get_batch_slice(n_features, batch_size)
    for i, (begin_row, end_row) in enumerate(zip(begin_ids, end_ids)):
        if verbose:
            print("Batch:", end_row, "/", n_features, flush=True)
        x1 = x[:, begin_row:end_row].T
        for j, (begin_col, end_col) in enumerate(zip(begin_ids, end_ids)):
            if i <= j:
                prod = x1 @ x[:, begin_col:end_col]
                xtx[begin_row:end_row, begin_col:end_col] += prod
    return xtx


def sum_xtx(
    xtx: np.ndarray,
    x: np.ndarray,
    n_features_threshold: int = 50000,
    n_batch: int = 10,
    verbose: bool = False,
):
    """Add x.T @ x to xtx."""
    n_features = x.shape[1]
    if verbose:
        print("Compute X.T @ X and X.T @ y", flush=True)
    if n_features < n_features_threshold:
        xtx = sum_array(xtx, x.T @ x)
    else:
        xtx = sum_large_xtx(xtx, x, n_batch=n_batch, verbose=verbose)
    return xtx


def symmetrize_xtx(xtx: np.ndarray, n_batch: int = 10):
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
