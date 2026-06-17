/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        Ref. Associated Legendre Polynomials and Spherical Harmonics
             Computation for Chemistry Applications (2014).
        Taweetham Limpanuparb and Josh Milthorpe.
        arXiv: 1410.1748 [physics.chem-ph]

        The current implementation is modified from the original algorithm.
        Derivative calculations in mlipkk_spherical_harmonics.cpp are revised.

*****************************************************************************/


#include "polymlp_spherical_harmonics.h"


SphericalHarmonics::SphericalHarmonics(const int lmax) : lmax_(lmax) {
    n_lm_half_ = (lmax + 1) * (lmax + 2) / 2;
}

SphericalHarmonics::~SphericalHarmonics(){}


void SphericalHarmonics::compute_ylm(
    const double costheta,
    const double cos_azimuthal,
    const double sin_azimuthal,
    vector1dc& ylm) const {

    /// List Spherical harmonics with m > 0.
    /// The order of (l, m) is (0, 0), (1, -1), (1, 0), (2, -2), ...
    ylm.resize(n_lm_half_);
    vector1d p;
    normalized_associated_legendre(costheta, p);
    for (int l = 0; l <= lmax_; ++l) {
        int idx = lm2i_negm(l, 0);
        ylm[idx] = p[lm2i(l, 0)] * 0.5 * M_SQRT2;
    }

    double c1 = 1.0, c2 = cos_azimuthal;  // cos(0 * phi) and cos(-1 * phi)
    double s1 = 0.0, s2 = -sin_azimuthal;  // sin(0 * phi) and sin(-1 * phi)
    const double tc = 2.0 * c2;
    double sign = -1;
    for (int mp = 1; mp <= lmax_; ++mp) {
        const double s = tc * s1 - s2;  // sin(mp * phi)
        const double c = tc * c1 - c2;  // cos(mp * phi)
        c2 = c1; c1 = c; s2 = s1; s1 = s;
        for (int l = mp; l <= lmax_; ++l) {
            int idx = lm2i_negm(l, -mp);
            const double tmp = sign * p[lm2i(l, mp)] * 0.5 * M_SQRT2;
            ylm[idx] = tmp * std::complex<double>(c, -s);
        }
        sign *= -1;
    }
}

void SphericalHarmonics::compute_ylm_der(
    const double costheta,
    const double cosphi,
    const double sinphi,
    const double r,
    vector1dc& ylm,
    vector1dc& ylm_dx,
    vector1dc& ylm_dy,
    vector1dc& ylm_dz) const {

    ylm.resize(n_lm_half_);
    ylm_dx.resize(n_lm_half_);
    ylm_dy.resize(n_lm_half_);
    ylm_dz.resize(n_lm_half_);

    vector1d p, q;
    normalized_associated_legendre(costheta, p, q);

    const double sintheta = sqrt(1.0 - costheta * costheta);
    const double invr = 1.0 / r;

    for (int l = 0; l <= lmax_; ++l) {
        int idx = lm2i_negm(l, 0);
        ylm[idx] = p[lm2i(l, 0)] * 0.5 * M_SQRT2;
        const double common = q[lm2i(l, 1)] * sintheta * invr * sqrt(0.5 * l * (l + 1));
        ylm_dx[idx] = common * costheta * cosphi;
        ylm_dy[idx] = common * costheta * sinphi;
        ylm_dz[idx] = -common * sintheta;
    }

    double c1 = 1.0, c2 = cosphi;  // cos(0 * phi) and cos(-1 * phi)
    double s1 = 0.0, s2 = -sinphi;  // sin(0 * phi) and sin(-1 * phi)
    const double tc = 2.0 * c2;
    double sign = -1.0;
    for (int mp = 1; mp <= lmax_; ++mp) {
        const double s = tc * s1 - s2;  // sin(mp * phi)
        const double c = tc * c1 - c2;  // cos(mp * phi)
        c2 = c1; c1 = c; s2 = s1; s1 = s;
        for (int l = mp; l <= lmax_; ++l) {
            int idx = lm2i_negm(l, -mp);
            const double tmp = sign * p[lm2i(l, mp)] * 0.5 * M_SQRT2;
            ylm[idx] = tmp * dc(c, -s);  // (l, m=-mp)

            const dc eimphi(c, s);
            const auto common = eimphi * 0.5 * M_SQRT2 * invr;

            double dtheta = mp * costheta * q[lm2i(l, mp)];
            if (mp != l) {
                dtheta += sqrt((l - mp) * (l + mp + 1))
                        * q[lm3i(l, mp + 1)] * sintheta;
            }
            const dc dphi(0.0, mp * q[lm2i(l, mp)]);

            ylm_dx[idx] = sign
                * std::conj(common * (dtheta * costheta * cosphi - dphi * sinphi));
            ylm_dy[idx] = sign
                * std::conj(common * (dtheta * costheta * sinphi + dphi * cosphi));
            ylm_dz[idx] = sign * std::conj(-common * dtheta * sintheta);
        }
        sign *= -1.0;
    }
}


void SphericalHarmonics::normalized_associated_legendre(
    const double costheta, vector1d& p) const {

    p.resize(n_lm_half_);

    const double sqrt_inv_2pi = 0.39894228040143267794;  // = sqrt(0.5 / M_PI)
    p[lm2i(0, 0)] = sqrt_inv_2pi;
    if (lmax_ == 0) return;

    const double SQRT3 = 1.7320508075688772935;
    p[lm2i(1, 0)] = costheta * SQRT3 * sqrt_inv_2pi;

    const double sintheta = sqrt(1.0 - costheta * costheta);
    const double SQRT3DIV2 = 1.2247448713915890491;
    p[lm2i(1, 1)] = - sintheta * SQRT3DIV2 * sqrt_inv_2pi;

    for (int l = 2; l <= lmax_; ++l) {
        double coeff1 = -sqrt(1.0 + 0.5 / l) * sintheta;
        p[lm2i(l, l)] = coeff1 * p[lm2i(l - 1, l - 1)];

        double coeff2 = sqrt(2.0 * (l - 1.0) + 3.0) * costheta;
        p[lm2i(l, l - 1)] = coeff2 * p[lm2i(l - 1, l - 1)];
    }

    for (int l = 2; l <= lmax_; ++l) {
        double ls = l * l;
        double lm1s = (l - 1) * (l - 1);
        for (int m = 0; m <= l - 2; ++m) {
            double ms = m * m;
            double alm = sqrt((4.0 * ls - 1.0) / (ls - ms));
            double blm = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
            p[lm2i(l, m)] =
                alm * (costheta * p[lm2i(l - 1, m)] + blm * p[lm2i(l - 2, m)]);
        }
    }
}


void SphericalHarmonics::normalized_associated_legendre(
    const double costheta,
    vector1d& p,
    vector1d& q) const {

    /// normalized associated Legendre polynomial P_{l}^{m} and P_{l}^{M} / sin_theta
    /// For m >= 0, Y_{l}^{m} = P_{l}^{m} exp^{im phi} / sqrt(2)
    p.resize(n_lm_half_);
    q.resize(n_lm_half_);

    const double sqrt_inv_2pi = 0.39894228040143267794;  // = sqrt(0.5 / M_PI)
    p[lm2i(0, 0)] = sqrt_inv_2pi;
    q[lm2i(0, 0)] = 0.0;
    if (lmax_ == 0) return;

    const double SQRT3 = 1.7320508075688772935;
    p[lm2i(1, 0)] = costheta * SQRT3 * sqrt_inv_2pi;
    q[lm2i(1, 0)] = 0.0;

    const double sintheta = sqrt(1.0 - costheta * costheta);
    const double SQRT3DIV2 = 1.2247448713915890491;
    p[lm2i(1, 1)] = - sintheta * SQRT3DIV2 * sqrt_inv_2pi;
    q[lm2i(1, 1)] = - SQRT3DIV2 * sqrt_inv_2pi;

    for (int l = 2; l <= lmax_; ++l) {
        double coeff1 = -sqrt(1.0 + 0.5 / l) * sintheta;
        p[lm2i(l, l)] = coeff1 * p[lm2i(l - 1, l - 1)];
        q[lm2i(l, l)] = coeff1 * q[lm2i(l - 1, l - 1)];

        double coeff2 = sqrt(2.0 * (l - 1.0) + 3.0) * costheta;
        p[lm2i(l, l - 1)] = coeff2 * p[lm2i(l - 1, l - 1)];
        q[lm2i(l, l - 1)] = coeff2 * q[lm2i(l - 1, l - 1)];
    }

    for (int l = 2; l <= lmax_; ++l) {
        double ls = l * l;
        double lm1s = (l - 1) * (l - 1);
        for (int m = 0; m <= l - 2; ++m) {
            double ms = m * m;
            double alm = sqrt((4.0 * ls - 1.0) / (ls - ms));
            double blm = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
            p[lm2i(l, m)] =
                alm * (costheta * p[lm2i(l - 1, m)] + blm * p[lm2i(l - 2, m)]);
            q[lm2i(l, m)] =
                alm * (costheta * q[lm2i(l - 1, m)] + blm * q[lm2i(l - 2, m)]);
        }
    }
}
