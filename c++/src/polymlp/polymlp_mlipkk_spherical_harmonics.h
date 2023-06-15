/* ----------------------------------------------------------------------
   Contributing author: Kohei Shinohara
------------------------------------------------------------------------- */

#ifndef MLIPKK_SPHERICAL_HARMONICS_H_
#define MLIPKK_SPHERICAL_HARMONICS_H_

#include <vector>
#include <complex>

/// Ref. Associated Legendre Polynomials and Spherical Harmonics Computation for Chemistry Applications (2014).
/// Taweetham Limpanuparb and Josh Milthorpe. arXiv: 1410.1748 [physics.chem-ph]
/// Some formulae are referred to Digital Library of Mathematical Functions (DLMF).
class SphericalHarmonics {
    using LMIdx = int;
    using LMInfoIdx = int;

    int lmax_;
    int n_lm_half_;
    int n_lm_all_;

    double* A_ = NULL;
    double* B_ = NULL;

public:
    SphericalHarmonics() = default;
    ~SphericalHarmonics();
    SphericalHarmonics(const int lmax);
    SphericalHarmonics(const SphericalHarmonics& other);
    SphericalHarmonics& operator=(const SphericalHarmonics& other);

    void compute_ylm(const double costheta, const double azimuthal, std::vector<std::complex<double>>& ylm) const;
    void compute_ylm_der(const double costheta, const double azimuthal, const double r,
                         std::vector<std::complex<double>>& ylm_dx, std::vector<std::complex<double>>& ylm_dy, std::vector<std::complex<double>>& ylm_dz) const;

    int get_lmax() const { return lmax_; };
    int get_n_lm_half() const { return n_lm_half_; };
    int get_n_lm_all() const { return n_lm_all_; };

private:
    void initAB(const int lmax);
    void compute_normalized_associated_legendre(const double costheta, std::vector<double>& p) const;
    void compute_normalized_associated_legendre_sintheta(const double costheta,
                                                         std::vector<double>& q) const;
    inline int lm2i(int l, int m) const { return (((l) * ((l) + 1)) / 2 + (m));  }

};

#endif // MLIPKK_SPHERICAL_HARMONICS_H_
