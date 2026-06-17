/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FUNCTIONS_INTERFACE
#define __POLYMLP_FUNCTIONS_INTERFACE

#include "polymlp_mlpcpp.h"
#include "polymlp_basis_function.h"
#include "polymlp_spherical_harmonics.h"
#include "polymlp_mlipkk_spherical_harmonics.h"

// Radial functions
void get_fn_(const double dis,
             const struct feature_params& fp,
             vector1d& fn);

void get_fn_(const double dis,
             const struct feature_params& fp,
             const vector2d& params,
             vector1d& fn);

void get_fn_(const double dis,
             const struct feature_params& fp,
             vector1d& fn,
             vector1d& fn_dr);

void get_fn_(const double dis,
             const struct feature_params& fp,
             const vector2d& params,
             vector1d& fn,
             vector1d& fn_dr);

// Spherical harmonics
void get_ylm_(const double x,
              const double y,
              const double z,
              const int lmax,
              vector1dc& ylm);


void get_ylm_(const double r,
              const double x,
              const double y,
              const double z,
              const int lmax,
              vector1dc& ylm,
              vector1dc& ylm_dx,
              vector1dc& ylm_dy,
              vector1dc& ylm_dz);

void get_ylm_(const double x,
              const double y,
              const double z,
              SphericalHarmonics& sh,
              vector1dc& ylm);


void get_ylm_(const double r,
              const double x,
              const double y,
              const double z,
              SphericalHarmonics& sh,
              vector1dc& ylm,
              vector1dc& ylm_dx,
              vector1dc& ylm_dy,
              vector1dc& ylm_dz);


void get_ylm_polar(const double polar,
              const double azimuthal,
              const int lmax,
              vector1dc& ylm);

void get_ylm_polar(const double r,
              const double polar,
              const double azimuthal,
              const int lmax,
              vector1dc& ylm,
              vector1dc& ylm_dx,
              vector1dc& ylm_dy,
              vector1dc& ylm_dz);


vector1d cartesian_to_spherical_(const vector1d& v);
vector1d cartesian_to_spherical_(const double x, const double y, const double z);

#endif
