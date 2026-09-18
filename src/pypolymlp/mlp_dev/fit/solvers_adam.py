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


def _add_regularization(
    error: NDArray, grad_trial: NDArray, coef: NDArray, coef0: NDArray, alpha: float
):
    """Add quadratic regularization term."""
    diff_coef = (coef - coef0) / np.abs(coef0)
    reg = alpha * (diff_coef @ diff_coef)
    grad_reg = alpha * diff_coef / np.abs(coef0)
    error += reg
    grad_trial += grad_reg
    return error, grad_trial


def solver_adam(
    x: NDArray,
    y: NDArray,
    coef0: Optional[NDArray] = None,
    alpha: float = 0.1,
    beta: float = 0.95,
    batch_size: int = 1000,
    gtol: float = 1e-2,
    n_epochs: int = 100,
    use_scales: bool = True,
    max_learning_rate: float = 1e-4,
    verbose: bool = False,
):
    """Estimate MLP coefficients using Adam.

    If alpha > 0, regularization term (alpha * || (w - w0) / w0 ||^2)
    is added to minimization function.

    Parameters
    ----------
    x: Predictor matrix, X.
    y: Observation vector, y.
    coef0: Initial coefficients.
    alpha: Magnitude parameter for regularization.
    beta: Parameter for defining gradient update.
    batch_size: Minibatch size.
    n_epochs: Number of epochs.
    """
    if verbose:
        print("Use Adam solver.", flush=True)
        print("conditions:", flush=True)
        print("- beta:       ", beta, flush=True)
        print("  batch_size: ", batch_size, flush=True)
        print("  gtol:       ", gtol, flush=True)
        print("  n_epochs:   ", n_epochs, flush=True)
    if alpha < 0:
        raise RuntimeError("Found negative alpha.")

    n_data, n_features = x.shape
    if coef0 is None:
        alpha = 0.0
        coef0 = np.zeros(n_features)
    else:
        coef0 = np.array(coef0)

    if use_scales:
        scales = np.std(x, axis=0)
        x /= scales
        coef0 *= scales

    coef = copy.deepcopy(coef0)

    beta2 = beta**2 / (beta**2 + (1 - beta) ** 2)
    if batch_size is None:
        batch_size = n_data // 5
    begin_batch, end_batch = _get_batch_slice(n_data, batch_size)
    eps_grad = gtol

    grad_prev, magn_prev = np.zeros(n_features), np.zeros(n_features)
    converge = False
    for i_epoch in range(n_epochs):
        if verbose:
            print("------", flush=True)
            print("Epoch:", i_epoch + 1, flush=True)

        rate = max(max_learning_rate / np.sqrt(i_epoch + 1), max_learning_rate * 1e-4)
        if verbose:
            print("- Learning rate:", "{:.8f}".format(rate), flush=True)

        for i_batch in _shuffle_batch_order(len(begin_batch)):
            begin, end = begin_batch[i_batch], end_batch[i_batch]
            x_batch, y_batch = x[begin:end], y[begin:end]
            n_data_batch = len(y_batch)

            error = x_batch @ coef - y_batch
            grad_trial = x_batch.T @ error
            error, grad_trial = _add_regularization(
                error, grad_trial, coef, coef0, alpha
            )
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

        if verbose:
            rmse = np.sqrt(np.average(np.square(x @ coef - y)))
            print("- RMSE:        ", "{:.7f}".format(rmse), flush=True)
            print("- Max gradient:", "{:.5e}".format(grad_max), flush=True)
            print("- Ave gradient:", "{:.5e}".format(grad_ave), flush=True)

        if converge:
            break

    if use_scales:
        x *= scales
        coef0 /= scales
        coef /= scales

    return coef
