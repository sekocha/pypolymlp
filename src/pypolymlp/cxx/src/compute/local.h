/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __LOCAL
#define __LOCAL

#include <cmath>

#include "mlpcpp.h"
#include "polymlp/polymlp_api.h"
#include "polymlp/polymlp_functions_interface.h"


class Local{

    int n_atom;

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
        const Polymlp& polymlp,
        const int type1,
        const vector2d& dis_a,
        const vector3d& diff_a,
        vector1dc& anlmtp
    );

    void compute_anlm_d(
        const Polymlp& polymlp,
        const int type1,
        const vector2d& dis_a,
        const vector3d& diff_a,
        const vector2i& atom2_a,
        vector1dc& anlmtp,
        vector2dc& anlmtp_dfx,
        vector2dc& anlmtp_dfy,
        vector2dc& anlmtp_dfz,
        vector2dc& anlmtp_ds
    );

    public:

    Local(const int n_atom);
    ~Local();

    void gtinv(
        const Polymlp& polymlp,
        const int type1,
        const vector2d& dis_a,
        const vector3d& diff_a,
        vector1d& dn
    );

    void gtinv_d(
        const Polymlp& polymlp,
        const int type1,
        const vector2d& dis_a,
        const vector3d& diff_a,
        const vector2i& atom2_a,
        vector1d& dn,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
    );
};

#endif
