/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __LOCAL_FAST
#define __LOCAL_FAST

#include <cmath>

#include "mlpcpp.h"
#include "polymlp/polymlp_mapping.h"
#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_products.h"
#include "compute/features.h"

class LocalFast{

    int n_atom, atom1, type1, n_fn, n_des, n_type, size_pair;
    struct feature_params fp;
    vector1i type2_array, type_pairs;

    void compute_linear_features(
        const vector1d& prod_anlmtc,
        const FunctionFeatures& features,
        vector1d& dn
    );

    void compute_linear_features_deriv(
        const vector1dc& prod_anlmtc_d,
        const FunctionFeatures& features,
        const vector2dc& anlmtc_dfx,
        const vector2dc& anlmtc_dfy,
        const vector2dc& anlmtc_dfz,
        const vector2dc& anlmtc_ds,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
    );

    void compute_anlm(
        const vector2d& dis_a,
        const vector3d& diff_a,
        const FunctionFeatures& features,
        vector1dc& anlm
    );
    void compute_anlm_d(
        const vector2d& dis_a,
        const vector3d& diff_a,
        const vector2i& atom2_a,
        const FunctionFeatures& features,
        vector1dc& anlm,
        vector2dc& anlm_dfx,
        vector2dc& anlm_dfy,
        vector2dc& anlm_dfz,
        vector2dc& anlm_ds
    );

    public:

    LocalFast();
    LocalFast(
        const int& n_atom_i,
        const int& atom1_i,
        const int& type1_i,
        const struct feature_params& fp_i,
        const FunctionFeatures& features
    );
    ~LocalFast();

    void pair(
        const vector2d& dis_a,
        const FunctionFeatures& features,
        vector1d& dn
    );
    void pair_d(
        const vector2d& dis_a,
        const vector3d& diff_a,
        const vector2i& atom2_a,
        const FunctionFeatures& features,
        vector1d& dn,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
    );

    void gtinv(
        const vector2d& dis_a,
        const vector3d& diff_a,
        const FunctionFeatures& features,
        vector1d& dn
    );
    void gtinv_d(
        const vector2d& dis_a,
        const vector3d& diff_a,
        const vector2i& atom2_a,
        const FunctionFeatures& features,
        vector1d& dn,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
    );
};

#endif
