"""API functions using C++ library."""

from pypolymlp.cxx.lib import libmlpcpp


def get_fn(r: float, params: list, cutoff: float):
    """Calculate spherical harmonics."""
    fp = libmlpcpp.FeatureParams()
    fp.pair_type = "gaussian"
    fp.cutoff = cutoff
    fn, fn_d = libmlpcpp.get_fn(r, fp, params)
    return (fn, fn_d)


def get_ylm(r: float, polar: float, azimuthal: float, lmax: int):
    """Calculate spherical harmonics."""
    ylm, ylm_dx, ylm_dy, ylm_dz = libmlpcpp.get_ylm(
        r=r,
        polar=polar,
        azimuthal=azimuthal,
        lmax=lmax,
    )
    return (ylm, ylm_dx, ylm_dy, ylm_dz)
