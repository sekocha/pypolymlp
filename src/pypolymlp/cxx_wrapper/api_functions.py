"""API functions using C++ library."""

import numpy as np

from pypolymlp.cxx.lib import libmlpcpp


def get_fn(r: float, params: list, cutoff: float):
    """Calculate spherical harmonics."""
    fp = libmlpcpp.FeatureParams()
    fp.pair_type = "gaussian"
    fp.cutoff = cutoff
    fn, fn_d = libmlpcpp.get_fn(r, fp, params)
    return (fn, fn_d)


def get_ylm(x: float, y: float, z: float, lmax: int):
    """Calculate spherical harmonics."""
    r = np.sqrt(x * x + y * y + z * z)
    ylm, ylm_dx, ylm_dy, ylm_dz = libmlpcpp.get_ylm(
        r=r,
        x=x,
        y=y,
        z=z,
        lmax=lmax,
    )
    return (ylm, ylm_dx, ylm_dy, ylm_dz)
