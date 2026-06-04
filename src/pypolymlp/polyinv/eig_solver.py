"""Eigenvalue problem solver."""

from numpy.typing import NDArray
from symfc.eig_solvers.eig_tools_core import eigh_projector
from symfc.eig_solvers.eig_tools_recursive import eigsh_projector_division


def eigh(
    p: NDArray,
    atol: float = 1e-7,
    rtol: float = 0.0,
    size_threshold: int = 500,
    log_level: int = 0,
):
    """Solve eigenvalue problem for matrix p."""
    verbose = log_level > 0
    if p.shape[0] < size_threshold:
        res = eigh_projector(p, atol=atol, rtol=rtol, verbose=verbose)
        return res.eigvecs

    res = eigsh_projector_division(
        p,
        atol=atol,
        rtol=rtol,
        return_cmplt=False,
        verbose=verbose,
    )
    eigvecs = res.eigvecs.recover()
    return eigvecs
