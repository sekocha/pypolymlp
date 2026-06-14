"""API functions using C++ library."""

from pypolymlp.cxx.lib import libmlpcpp


def get_gtinv_attrs(order: int, lmax: tuple, version: int):
    """Return invariant information."""
    rgi = libmlpcpp.Readgtinv(order, lmax, version)
    lm_seq = rgi.get_lm_seq()
    l_comb = rgi.get_l_comb()
    lm_coeffs = rgi.get_lm_coeffs()
    return (l_comb, lm_seq, lm_coeffs)
