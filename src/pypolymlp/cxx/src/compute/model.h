/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __MODEL
#define __MODEL

#include "mlpcpp.h"
#include "polymlp/polymlp_api.h"
#include "compute/local.h"
#include "compute/local_pair.h"


class Model {

    PolymlpAPI polymlp;

    int n_atom;
    bool force;
    vector1i types;

    vector1d xe_sum;
    vector2d xf_sum, xs_sum;

    void pair(
        const vector3d& dis_array,
        const vector4d& diff_array,
        const vector3i& atom2_array
    );

    void gtinv(
        const vector3d& dis_array,
        const vector4d& diff_array,
        const vector3i& atom2_array
    );

    void model_polynomial(
        const vector1d& de,
        const vector2d& dfx,
        const vector2d& dfy,
        const vector2d& dfz,
        const vector2d& ds,
        const int type1
    );

    void model_order1(
        const PolynomialTerm& term,
        const vector1d& de,
        const vector2d& dfx,
        const vector2d& dfy,
        const vector2d& dfz,
        const vector2d& ds
    );
    void model_order2(
        const PolynomialTerm& term,
        const vector1d& de,
        const vector2d& dfx,
        const vector2d& dfy,
        const vector2d& dfz,
        const vector2d& ds
    );
    void model_order3(
        const PolynomialTerm& term,
        const vector1d& de,
        const vector2d& dfx,
        const vector2d& dfy,
        const vector2d& dfz,
        const vector2d& ds
    );

    public:

    Model();
    Model(
        const vector3d& dis_array_all,
        const vector4d& diff_array_all,
        const vector3i& atom2_array_all,
        const vector1i& types_i,
        const struct feature_params& fp
    );
    ~Model();

    const vector1d& get_xe_sum() const;
    const vector2d& get_xf_sum() const;
    const vector2d& get_xs_sum() const;

};

#endif
