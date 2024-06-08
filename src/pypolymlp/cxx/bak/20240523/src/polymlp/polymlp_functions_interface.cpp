/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_functions_interface.h"

void get_fn_(const double& dis,
             const struct feature_params& fp,
             vector1d& fn){

    double fc = cosine_cutoff_function(dis, fp.cutoff);

    fn.resize(fp.params.size());
    if (fp.pair_type == "gaussian"){
        for (size_t n = 0; n < fp.params.size(); ++n){
            fn[n] = gauss(dis, fp.params[n][0], fp.params[n][1]) * fc;
        }
    }
    /*
    else if (fp.pair_type == "sph_bessel"){
        for (int n = 0; n < fp.params.size(); ++n){
            fn[n] = sph_bessel(dis, fp.params[n][0], fp.params[n][1]) * fc;
        }
    }
    */
}

void get_fn_(const double& dis,
             const struct feature_params& fp,
             vector1d& fn,
             vector1d& fn_dr){

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
    /*
    else if (fp.pair_type == "sph_bessel"){
        for (int n = 0; n < fp.params.size(); ++n){
            sph_bessel_d(dis, fp.params[n][0], fp.params[n][1],
                         fn_val, fn_dr_val);
            fn[n] = fn_val * fc;
            fn_dr[n] = fn_dr_val * fc + fn_val * fc_dr;
        }
    }
    */
}

void get_ylm_(const double polar,
              const double azimuthal,
              const int lmax,
              vector1dc& ylm){

    SphericalHarmonics sh(lmax);
    sh.compute_ylm(cos(polar), azimuthal, ylm);
}

void get_ylm_(const double r,
              const double polar,
              const double azimuthal,
              const int lmax,
              vector1dc& ylm,
              vector1dc& ylm_dx,
              vector1dc& ylm_dy,
              vector1dc& ylm_dz){

    SphericalHarmonics sh(lmax);
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

/*
vector1d cartesian_to_spherical_(const vector1d& v){

    bg::model::point<long double,3,bg::cs::cartesian> p1(v[0], v[1], v[2]);
    bg::model::point<long double,3,bg::cs::spherical<bg::radian> > p2;
    bg::transform(p1, p2);
    return vector1d {static_cast<double>(bg::get<1>(p2)),
        static_cast<double>(bg::get<0>(p2))}; // theta, phi
}
*/
