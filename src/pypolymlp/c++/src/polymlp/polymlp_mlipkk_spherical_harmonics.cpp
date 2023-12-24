/* ----------------------------------------------------------------------
   Contributing author: Kohei Shinohara
------------------------------------------------------------------------- */

#include "polymlp_mlipkk_spherical_harmonics.h"

#include <complex>
#include <cmath>
#include <cassert>

#include <iostream>

SphericalHarmonics::~SphericalHarmonics() {
    if (A_ != NULL) {
        delete [] A_;
        delete [] B_;
    }
}

SphericalHarmonics& SphericalHarmonics::operator=(const SphericalHarmonics& other) {
    if (this != &other) {
        const int lmax = other.get_lmax();
        n_lm_half_ = other.get_n_lm_half();
        n_lm_all_ = other.get_n_lm_all();

        A_ = new double [n_lm_half_];
        B_ = new double [n_lm_half_];

        initAB(lmax);
    }
    return *this;
}

SphericalHarmonics::SphericalHarmonics(const SphericalHarmonics& other) : lmax_(other.get_lmax()) {
    *this = other;
}

SphericalHarmonics::SphericalHarmonics(const int lmax) : lmax_(lmax) {
    n_lm_half_ = (lmax + 1) * (lmax + 2) / 2;
    n_lm_all_ = (lmax + 1) * (lmax + 1);

    A_ = new double [n_lm_half_];
    B_ = new double [n_lm_half_];

    initAB(lmax);
}

void SphericalHarmonics::initAB(const int lmax) {
    for (int l = 2; l <= lmax; ++l) {
        double ls = l * l;
        double lm1s = (l - 1) * (l - 1);
        for (int m = 0; m <= l - 2; ++m) {
            double ms = m * m;
            A_[lm2i(l, m)] = sqrt((4.0 * ls - 1.0) / (ls - ms));
            B_[lm2i(l, m)] = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
        }
    }
}

/// @brief spherical harmonics for (l, m) = (0, 0), (1, -1), (1, 0), (2, -2), ..., (lmax, 0)
/// Be careful returned list does not contain m > 0
/// @param[in] costheta cosine of polar angle
/// @param[in] azimuthal [0, 2 * pi)
/// @param[out] ylm
void SphericalHarmonics::compute_ylm(const double costheta, const double azimuthal,
                                     std::vector<std::complex<double>>& ylm) const
{
    ylm.resize(n_lm_half_);
    std::vector<double> p;
    compute_normalized_associated_legendre(costheta, p);
    for (int l = 0; l <= lmax_; ++l) {
        ylm[lm2i(l, l)] = p[lm2i(l, 0)] * 0.5 * M_SQRT2;  // (l, m=0)
    }

    double c1 = 1.0, c2 = cos(azimuthal);  // cos(0 * phi) and cos(-1 * phi)
    double s1 = 0.0, s2 = -sin(azimuthal);  // sin(0 * phi) and sin(-1 * phi)
    const double tc = 2.0 * c2;
    double sign = -1;
    for (int mp = 1; mp <= lmax_; ++mp) {
        const double s = tc * s1 - s2;  // sin(mp * phi)
        const double c = tc * c1 - c2;  // cos(mp * phi)
        c2 = c1; c1 = c; s2 = s1; s1 = s;
        for (int l = mp; l <= lmax_; ++l) {
            const double tmp = sign * p[lm2i(l, mp)] * 0.5 * M_SQRT2;
            ylm[lm2i(l, l - mp)] = tmp * std::complex<double>(c, -s);  // (l, m=-mp)
        }
        sign *= -1;
    }
}

/// @brief spherical harmonics and its derivative with cartesian coordinates
/// Be careful returned list does not contain m > 0
/// @param[in] costheta cosine of polar angle
/// @param[in] azimuthal [0, 2 * pi)
/// @param[out] ylm_dx
/// @param[out] ylm_dy
/// @param[out] ylm_dz
void SphericalHarmonics::compute_ylm_der(const double costheta, const double azimuthal, const double r,
                                         std::vector<std::complex<double>>& ylm_dx, std::vector<std::complex<double>>& ylm_dy, std::vector<std::complex<double>>& ylm_dz) const
{
    ylm_dx.resize(n_lm_half_);
    ylm_dy.resize(n_lm_half_);
    ylm_dz.resize(n_lm_half_);

    std::vector<double> q;
    compute_normalized_associated_legendre_sintheta(costheta, q);

    const double sintheta = sqrt(1.0 - costheta * costheta);
    const double cosphi = cos(azimuthal);
    const double sinphi = sin(azimuthal);

    double c1 = 1.0, c2 = cosphi;  // cos(0 * phi) and cos(-1 * phi)
    double s1 = 0.0, s2 = -sinphi;  // sin(0 * phi) and sin(-1 * phi)
    const double tc = 2.0 * c2;
    const double invr = 1.0 / r;

    // (l, 0)
    for (int l = 0; l <= lmax_; ++l) {
        const double common = q[lm2i(l, 1)] * sintheta * invr * sqrt(0.5 * l * (l + 1));
        ylm_dx[lm2i(l, l)] = common * costheta * cosphi;
        ylm_dy[lm2i(l, l)] = common * costheta * sinphi;
        ylm_dz[lm2i(l, l)] = -common * sintheta;
    }

    double sign = -1.0;
    for (int mp = 1; mp <= lmax_; ++mp) {
        const double s = tc * s1 - s2;  // sin(mp * phi)
        const double c = tc * c1 - c2;  // cos(mp * phi)
        c2 = c1; c1 = c; s2 = s1; s1 = s;
        for (int l = mp; l <= lmax_; ++l) {
            const std::complex<double> eimphi(c, s);
            const auto common = eimphi * 0.5 * M_SQRT2 * invr;

            double dtheta = mp * costheta * q[lm2i(l, mp)];
            if (mp != l) {
                dtheta += sqrt((l - mp) * (l + mp + 1)) * q[lm2i(l, mp + 1)] * sintheta;  // TODO: reuse p[]
            }
            const std::complex<double> dphi(0.0, mp * q[lm2i(l, mp)]);

            ylm_dx[lm2i(l, l - mp)] = sign * std::conj(common * (dtheta * costheta * cosphi - dphi * sinphi));
            ylm_dy[lm2i(l, l - mp)] = sign * std::conj(common * (dtheta * costheta * sinphi + dphi * cosphi));
            ylm_dz[lm2i(l, l - mp)] = sign * std::conj(-common * dtheta * sintheta);
        }
        sign *= -1.0;
    }
}

/// normalized associated Legendre polynomial P_{l}^{m}
/// For m >= 0, Y_{l}^{m} = P_{l}^{m} exp^{im phi} / sqrt(2)
void SphericalHarmonics::compute_normalized_associated_legendre(const double costheta, std::vector<double>& p) const {
    p.resize(n_lm_half_);
    const double sintheta = sqrt(1.0 - costheta * costheta);

    double tmp = 0.39894228040143267794;  // = sqrt(0.5 / M_PI)
    p[lm2i(0, 0)] = tmp;
    if (lmax_ == 0) {
        return;
    }

    const double SQRT3 = 1.7320508075688772935;
    p[lm2i(1, 0)] = costheta * SQRT3 * tmp;
    const double SQRT3DIV2 = -1.2247448713915890491;
    tmp *= SQRT3DIV2 * sintheta;
    p[lm2i(1, 1)] = tmp;

    for (int l = 2; l <= lmax_; ++l) {
        for (int m = 0; m <= l - 2; ++m) {
            // DLMF 14.10.3
            p[lm2i(l, m)] = A_[lm2i(l, m)]
                            * (costheta * p[lm2i(l - 1, m)]
                               + B_[lm2i(l, m)] * p[lm2i(l - 2, m)]);
        }
        // DLMF
        p[lm2i(l, l - 1)] = costheta * sqrt(2.0 * (l - 1.0) + 3.0) * tmp;
        tmp *= -sqrt(1.0 + 0.5 / l) * sintheta;
        // DLMF 14.7.15
        p[lm2i(l, l)] = tmp;
    }
}

/// normalized associated Legendre polynomial divied by sintheta, P_{l}^{m}/sintheta
void SphericalHarmonics::compute_normalized_associated_legendre_sintheta(const double costheta,
                                                                         std::vector<double>& q) const
{
    q.resize(n_lm_half_, 0.0);
    if (lmax_ == 0) {
        return;
    }

    double tmp = -0.48860251190291992263;  // -sqrt(3 / (4 * M_PI))

    q[lm2i(1, 1)] = tmp;

    const double sintheta = sqrt(1.0 - costheta * costheta);
    for (int l = 2; l <= lmax_; ++l) {
        for (int m = 1; m <= l - 2; ++m) {
            // DLMF 14.10.3
            q[lm2i(l, m)] = A_[lm2i(l, m)]
                            * (costheta * q[lm2i(l - 1, m)]
                               + B_[lm2i(l, m)] * q[lm2i(l - 2, m)]);
        }
        // DLMF
        q[lm2i(l, l - 1)] = costheta * sqrt(2.0 * (l - 1.0) + 3.0) * tmp;
        tmp *= -sqrt(1.0 + 0.5 / l) * sintheta;
        // DLMF 14.7.15
        q[lm2i(l, l)] = tmp;
    }
}
