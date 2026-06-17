/* ----------------------------------------------------------------------
   Contributing author: Kohei Shinohara
------------------------------------------------------------------------- */

#ifndef MLIPKK_SPHERICAL_HARMONICS_H_
#define MLIPKK_SPHERICAL_HARMONICS_H_

#include "polymlp_mlpcpp.h"

/// Ref. Associated Legendre Polynomials and
///      Spherical Harmonics Computation for Chemistry Applications (2014).
/// Taweetham Limpanuparb and Josh Milthorpe. arXiv: 1410.1748 [physics.chem-ph]
/// Some formulae are referred to Digital Library of Mathematical Functions (DLMF).
class SphericalHarmonicsDep {

    vector1d A_, B_;
    int lmax_;
    int n_lm_half_;

    inline int lm2i(int l, int m) const {
        return (l * (l + 1) / 2) + m;
    }
    inline int lm2i_negm(int l, int m) const {
        return lm2i(l, m) + l;
    }

    void initAB(const int lmax);

    void normalized_associated_legendre(
        const double costheta,
        vector1d& p,
        vector1d& q) const;

    void compute_normalized_associated_legendre(
        const double costheta, std::vector<double>& p) const;
    void compute_normalized_associated_legendre_sintheta(
        const double costheta, std::vector<double>& q) const;

    public:

    SphericalHarmonicsDep(const int lmax);
    ~SphericalHarmonicsDep();

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

    void compute_ylm(
        const double costheta,
        const double azimuthal,
        std::vector<std::complex<double>>& ylm) const;

    void compute_ylm_der(
        const double costheta,
        const double azimuthal,
        const double r,
        std::vector<std::complex<double>>& ylm_dx,
        std::vector<std::complex<double>>& ylm_dy,
        std::vector<std::complex<double>>& ylm_dz) const;
    private:
};

#endif // MLIPKK_SPHERICAL_HARMONICS_H_
