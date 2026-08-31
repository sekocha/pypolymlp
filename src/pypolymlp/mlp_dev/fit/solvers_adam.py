"""Solvers for Adam."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _get_batch_slice(n_data: int, batch_size: int) -> tuple[list[int], list[int]]:
    """Calculate slice indices for a given batch size."""
    begin_batch = list(range(0, n_data, batch_size))
    if len(begin_batch) > 1:
        end_batch = list(begin_batch[1:]) + [n_data]
        if (end_batch[-1] - end_batch[-2]) < batch_size // 5:
            end_batch[-2] = end_batch[-1]
            begin_batch = begin_batch[:-1]
            end_batch = end_batch[:-1]
    else:
        end_batch = [n_data]
    return begin_batch, end_batch


def _shuffle_batch_order(batch_size: int):
    """Return shuffled batch order."""
    order = np.arange(batch_size)
    np.random.shuffle(order)
    return order


def _update_coefs_adam(
    coefs: NDArray,
    grad: NDArray,
    magn: NDArray,
    rate: float,
    eps_grad: float = 1e-6,
):
    """Update coefficients using gradients in Adam."""
    magn_sqrt = np.sqrt(magn)
    magn_sqrt[magn_sqrt < eps_grad] = np.inf
    coefs -= rate * grad / magn_sqrt
    return coefs


def _update_gradients_adam(
    grad_current: NDArray,
    grad_prev: NDArray,
    magn_prev: NDArray,
    beta: float,
    beta2: float,
):
    """Update gradients in Adam."""
    grad = beta * grad_prev + (1 - beta) * grad_current
    magn = beta2 * magn_prev + (1 - beta2) * (grad_current**2)
    return grad, magn


def _calc_gradient_stats(grad: NDArray):
    """Calculate average and maximum gradients."""
    grad_abs = np.abs(grad)
    grad_ave = np.average(grad_abs)
    grad_max = np.max(grad_abs)
    return grad_ave, grad_max


def solver_adam(
    x: np.ndarray,
    y: np.ndarray,
    coef0: Optional[np.ndarray] = None,
    beta: float = 0.95,
    batch_size: int = 100,
    gtol: float = 1e-2,
    n_epochs: int = 100,
    verbose: bool = False,
):
    """Estimate MLP coefficients using Adam."""
    if verbose:
        print("Use Adam solver.", flush=True)
        print("conditions:", flush=True)
        print("- beta:       ", beta, flush=True)
        print("  batch_size: ", batch_size, flush=True)
        print("  gtol:       ", gtol, flush=True)
        print("  n_epochs:   ", n_epochs, flush=True)

    n_data, n_features = x.shape
    beta2 = beta**2 / (beta**2 + (1 - beta) ** 2)
    begin_batch, end_batch = _get_batch_slice(n_data, batch_size)
    eps_grad = gtol

    coef = np.zeros(n_features) if coef0 is None else copy.deepcopy(coef0)
    grad_prev, magn_prev = np.zeros(n_features), np.zeros(n_features)
    converge = False
    for i_epoch in range(n_epochs):
        if verbose:
            print("------", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        rate = max(100 / np.sqrt(i_epoch + 1), 1e-4)
        if verbose:
            print("- Learning rate:", "{:.5f}".format(rate), flush=True)

        for i_batch in _shuffle_batch_order(len(begin_batch)):
            begin, end = begin_batch[i_batch], end_batch[i_batch]
            x_batch, y_batch = x[begin:end], y[begin:end]
            n_data_batch = len(y_batch)

            error = x_batch @ coef - y_batch
            grad_trial = x_batch.T @ error
            grad_trial /= n_data_batch

            grad, magn = _update_gradients_adam(
                grad_trial, grad_prev, magn_prev, beta, beta2
            )
            grad_ave, grad_max = _calc_gradient_stats(grad)
            if grad_ave < gtol and grad_max < gtol * 10:
                converge = True
                break

            coef = _update_coefs_adam(coef, grad, magn, rate, eps_grad)
            grad_prev, magn_prev = grad, magn

        # if verbose:
        #    error_all = np.array(error_all)
        #    rmse_forces = np.sqrt(np.mean(error_all**2))
        #    print("- Time:              ", "{:.3f}".format(t2 - t1), "s", flush=True)
        #    print("- RMSE (Force):      ", "{:.5e}".format(rmse_forces), flush=True)
        #    print("- Max gradient (FC2):", "{:.5e}".format(grad_max), flush=True)
        #    print("- Ave gradient (FC2):", "{:.5e}".format(grad_ave), flush=True)

        if converge:
            break
    print(coef)

    return coef
