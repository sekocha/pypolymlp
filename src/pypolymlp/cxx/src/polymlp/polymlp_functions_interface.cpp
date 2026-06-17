/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_functions_interface.h"


void get_fn_(const double dis,
             const struct feature_params& fp,
             const vector2d& params,
             vector1d& fn){

    double fc = cosine_cutoff_function(dis, fp.cutoff);

    fn.resize(params.size());
    if (fp.pair_type == "gaussian"){
        for (size_t n = 0; n < params.size(); ++n){
            fn[n] = gauss(dis, params[n][0], params[n][1]) * fc;
        }
    }
}

void get_fn_(const double dis,
             const struct feature_params& fp,
             vector1d& fn){

    /*** deprecated ***/
    double fc = cosine_cutoff_function(dis, fp.cutoff);

    fn.resize(fp.params.size());
    if (fp.pair_type == "gaussian"){
        for (size_t n = 0; n < fp.params.size(); ++n){
            fn[n] = gauss(dis, fp.params[n][0], fp.params[n][1]) * fc;
        }
    }
}

void get_fn_(const double dis,
             const struct feature_params& fp,
             const vector2d& params,
             vector1d& fn,
             vector1d& fn_dr){

    double fn_val, fn_dr_val;
    const double fc = cosine_cutoff_function(dis, fp.cutoff);
    const double fc_dr = cosine_cutoff_function_d(dis, fp.cutoff);

    fn.resize(params.size());
    fn_dr.resize(params.size());
    if (fp.pair_type == "gaussian"){
        for (size_t n = 0; n < params.size(); ++n){
            gauss_d(dis, params[n][0], params[n][1], fn_val, fn_dr_val);
            fn[n] = fn_val * fc;
            fn_dr[n] = fn_dr_val * fc + fn_val * fc_dr;
        }
    }
}


void get_fn_(const double dis,
             const struct feature_params& fp,
             vector1d& fn,
             vector1d& fn_dr){

    /*** deprecated ***/
    double fn_val, fn_dr_val;
    const double fc = cosine_cutoff_function(dis, fp.cutoff);
    const double fc_dr = cosine_cutoff_function_d(dis, fp.cutoff);

    fn.resize(fp.params.size());
    fn_dr.resize(fp.params.size());
    if (fp.pair_type == "gaussian"){
        for (size_t n = 0; n < fp.params.size(); ++n){
            gauss_d(dis, fp.params[n][0], fp.params[n][1], fn_val, fn_dr_val);
            fn[n] = fn_val * fc;
            fn_dr[n] = fn_dr_val * fc + fn_val * fc_dr;
        }
    }
}


void get_ylm_(
    const double x,
    const double y,
    const double z,
    const int lmax,
    vector1dc& ylm){

    double r = std::sqrt(x*x + y*y + z*z);
    double cos_theta = z / r;

    double cos_azimuthal, sin_azimuthal;
    double rho = std::hypot(x, y);
    if (rho > 0.0) {
        cos_azimuthal = x / rho;
        sin_azimuthal = y / rho;
    }
    else {
        cos_azimuthal = 1.0;
        sin_azimuthal = 0.0;
    }
    SphericalHarmonics sh(lmax);
    sh.compute_ylm(cos_theta, cos_azimuthal, sin_azimuthal, ylm);
}


void get_ylm_(
    const double r,
    const double x,
    const double y,
    const double z,
    const int lmax,
    vector1dc& ylm,
    vector1dc& ylm_dx,
    vector1dc& ylm_dy,
    vector1dc& ylm_dz){

    double cos_theta = z / r;
    double cos_azimuthal, sin_azimuthal;
    double rho = std::hypot(x, y);
    if (rho > 0.0) {
        cos_azimuthal = x / rho;
        sin_azimuthal = y / rho;
    }
    else {
        cos_azimuthal = 1.0;
        sin_azimuthal = 0.0;
    }

    SphericalHarmonics sh(lmax);
    sh.compute_ylm_der(
        cos_theta, cos_azimuthal, sin_azimuthal, r,
        ylm, ylm_dx, ylm_dy, ylm_dz
    );
}


void get_ylm_polar(
    const double polar,
    const double azimuthal,
    const int lmax,
    vector1dc& ylm){

    /*** deprecated ***/
    SphericalHarmonicsDep sh(lmax);
    sh.compute_ylm(cos(polar), azimuthal, ylm);
}

void get_ylm_polar(
    const double r,
    const double polar,
    const double azimuthal,
    const int lmax,
    vector1dc& ylm,
    vector1dc& ylm_dx,
    vector1dc& ylm_dy,
    vector1dc& ylm_dz){

    /*** deprecated ***/
    SphericalHarmonicsDep sh(lmax);
    sh.compute_ylm(cos(polar), azimuthal, ylm);
    sh.compute_ylm_der(cos(polar), azimuthal, r, ylm_dx, ylm_dy, ylm_dz);
}

vector1d cartesian_to_spherical_(const vector1d& v){

    double r, theta, phi;
    r = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    theta = std::acos(v[2] / r);
    phi = std::atan2(v[1], v[0]);
    return vector1d {theta, phi};
}

vector1d cartesian_to_spherical_(const double x, const double y, const double z){

    double r, theta, phi;
    r = sqrt(x*x + y*y + z*z);
    theta = std::acos(z / r);
    phi = std::atan2(y, x);
    return vector1d {theta, phi};
}
