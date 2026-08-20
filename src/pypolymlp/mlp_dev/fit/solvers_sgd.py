"""Solvers for stochastic gradient descent."""

from typing import Optional

import numpy as np


def _func(x, xty, coef, grad, alpha, beta, learning_rate):
    grad_add = x.T @ (x @ coef) + alpha * coef - xty
    grad_new = beta * grad + (1 - beta) * (-grad_add)
    coef += learning_rate * grad_new
    grad = grad_new
    return coef, grad


def _func_adam(
    x,
    xty,
    coef,
    grad=None,
    sq_grad=None,
    alpha=0.001,
    beta1=0.9,
    beta2=0.999,
    learning_rate=1e-2,
):
    """Function for Adam."""
    grad_add = x.T @ (x @ coef) + alpha * coef - xty
    if grad is None:
        grad_new = grad_add
    else:
        grad_new = beta1 * grad + (1 - beta1) * grad_add

    sq_grad_add = np.square(grad_add)
    if sq_grad is None:
        sq_grad_new = sq_grad_add
    else:
        sq_grad_new = beta2 * sq_grad + (1 - beta2) * sq_grad_add

    coef -= learning_rate * (grad_new / np.sqrt(sq_grad_new))
    grad = grad_new
    sq_grad = sq_grad_new
    return coef, grad, sq_grad


def solver_sgd(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    learning_rate: float = 1e-1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    coef0: Optional[np.ndarray] = None,
    gtol: float = 1e-1,
    max_iter: int = 100000,
    verbose: bool = False,
):
    """Estimate MLP coefficients using stochastic gradient descent."""
    # TODO: learning rate should be automatically determined.

    if verbose:
        print("Use SGD solver.", flush=True)
        print("- alpha:", alpha, flush=True)

    xty = x.T @ y
    if coef0 is None:
        coef = np.zeros(x.shape[1])
    else:
        coef = coef0

    norm = 1e10
    print("*********")
    grad, sq_grad = None, None
    for i in range(max_iter):
        if norm < 1e-0:
            break
        coef, grad, sq_grad = _func_adam(
            x,
            xty,
            coef,
            grad,
            sq_grad,
            alpha,
            beta1,
            beta2,
            learning_rate=0.3,
        )
        norm_new = np.linalg.norm(grad) / x.shape[1]
        norm = norm_new
        if i % 100 == 0:
            print(norm, grad)

    print("*********")
    grad, sq_grad = None, None
    for i in range(max_iter):
        if norm < 1e-1:
            break
        coef, grad, sq_grad = _func_adam(
            x, xty, coef, grad, sq_grad, alpha, beta1, beta2, learning_rate=0.01
        )
        norm_new = np.linalg.norm(grad) / x.shape[1]
        norm = norm_new
        if i % 100 == 0:
            print(norm, grad)

    print("*********")
    grad, sq_grad = None, None
    for i in range(max_iter):
        if norm < 1e-2:
            break
        coef, grad, sq_grad = _func_adam(
            x,
            xty,
            coef,
            grad,
            sq_grad,
            alpha,
            beta1,
            beta2,
            learning_rate=1e-3,
        )
        norm_new = np.linalg.norm(grad) / x.shape[1]
        norm = norm_new
        if i % 100 == 0:
            print(norm, grad)

    print("*********")
    grad, sq_grad = None, None
    for i in range(max_iter):
        if norm < 1e-3:
            break
        coef, grad, sq_grad = _func_adam(
            x,
            xty,
            coef,
            grad,
            sq_grad,
            alpha,
            beta1,
            beta2,
            learning_rate=1e-4,
        )
        norm_new = np.linalg.norm(grad) / x.shape[1]
        norm = norm_new
        if i % 100 == 0:
            print(norm, grad)

    #     norm = 1e10
    #     for i in range(1000):
    #         coef, grad = _func(x, xty, coef, grad, alpha, beta, 1e-8)
    #         norm = np.linalg.norm(grad) / x.shape[1]
    #         print(norm, grad)
    #
    #     for i in range(max_iter):
    #         if i > 0 and norm < gtol:
    #             break
    #         coef, grad = _func(
    #             x, xty, coef, grad, alpha, beta, learning_rate
    #         )
    #         norm = np.linalg.norm(grad) / x.shape[1]
    #         print(norm, grad)
    #

    return coef
