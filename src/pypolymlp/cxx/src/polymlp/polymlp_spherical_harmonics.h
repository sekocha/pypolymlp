/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        Ref. Associated Legendre Polynomials and Spherical Harmonics
             Computation for Chemistry Applications (2014).
        Taweetham Limpanuparb and Josh Milthorpe.
        arXiv: 1410.1748 [physics.chem-ph]

        The current implementation is modified from the original algorithm.

*****************************************************************************/


#ifndef POLYMLP_SPHERICAL_HARMONICS
#define POLYMLP_SPHERICAL_HARMONICS

#include "polymlp_mlpcpp.h"


class SphericalHarmonics {

    int lmax_;
    int n_lm_half_;

    inline int lm2i(int l, int m) const {
        return (l * (l + 1) / 2) + m;
    }
    inline int lm2i_negm(int l, int m) const {
        return lm2i(l, m) + l;
    }

    void normalized_associated_legendre(
        const double costheta,
        vector1d& p) const;
    void normalized_associated_legendre(
        const double costheta,
        vector1d& p,
        vector1d& q) const;

    public:

    SphericalHarmonics(const int lmax);
    ~SphericalHarmonics();

    void compute_ylm(
        const double costheta,
        const double cos_azimuthal,
        const double sin_azimuthal,
        vector1dc& ylm) const;

    void compute_ylm_der(
        const double costheta,
        const double cosphi,
        const double sinphi,
        const double r,
        vector1dc& ylm,
        vector1dc& ylm_dx,
        vector1dc& ylm_dy,
        vector1dc& ylm_dz) const;
};

#endif
