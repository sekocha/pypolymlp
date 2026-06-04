"""Eigenvalue problem solver."""

from numpy.typing import NDArray
from scipy.sparse import csr_array
from symfc.eig_solvers.eig_tools_core import eigh_projector
from symfc.eig_solvers.eig_tools_recursive import eigsh_projector_division
from symfc.eig_solvers.eig_tools_sparse import CompressionProjector


def eigsh(p: csr_array, log_level: int = 0):
    """Solve eigenvalue problem for given csr_array."""
    compr = CompressionProjector(p)
    cp = compr.compressed_projector.toarray()
    eigvecs = eigh(cp, log_level=log_level)
    return eigvecs


def eigh(
    p: NDArray,
    atol: float = 1e-8,
    rtol: float = 0.0,
    size_threshold: int = 500,
    log_level: int = 0,
):
    """Solve eigenvalue problem for matrix p."""
    verbose = log_level > 0
    if p.shape[0] < size_threshold:
        res = eigh_projector(p, atol=atol, rtol=rtol, verbose=verbose)
        eigvecs = res.eigvecs
    else:
        res = eigsh_projector_division(
            p,
            atol=atol,
            rtol=rtol,
            return_cmplt=False,
            verbose=verbose,
        )
        eigvecs = res.eigvecs.recover()
    return eigvecs
