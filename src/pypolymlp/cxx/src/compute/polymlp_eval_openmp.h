/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_EVAL_OPENMP
#define __POLYMLP_EVAL_OPENMP

#include "mlpcpp.h"

#include "polymlp/polymlp_api.h"
#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_products.h"
#include "compute/polymlp_eval.h"


class PolymlpEvalOpenMP {

    PolymlpAPI polymlp_api;

    /* for feature_type = pair */
    void compute_antp(
        const vector1i& types,
        const vector2i& neighbor_half,
        const vector3d& neighbor_diff,
        vector2d& antp
    );

    void compute_sum_of_prod_antp(
        const vector1i& types,
        const vector2d& antp,
        vector2d& prod_sum_e,
        vector2d& prod_sum_f
    );

    void eval_pair(
        const vector1i& types,
        const vector2i& neighbor_half,
        const vector3d& neighbor_diff,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    /* for feature_type = gtinv */
    void compute_anlmtp(
        const vector1i& types,
        const vector2i& neighbor_half,
        const vector3d& neighbor_diff,
        vector2dc& anlmtp
    );

    void compute_sum_of_prod_anlmtp(
        const vector1i& types,
        const vector2dc& anlmtp,
        vector2dc& prod_sum_e,
        vector2dc& prod_sum_f
    );

    void eval_gtinv(
        const vector1i& types,
        const vector2i& neighbor_half,
        const vector3d& neighbor_diff,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    void collect_properties(
        const vector2d& e_array,
        const vector2d& fx_array,
        const vector2d& fy_array,
        const vector2d& fz_array,
        const vector2i& neighbor_half,
        const vector3d& neighbor_diff,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    public:

    PolymlpEvalOpenMP();
    PolymlpEvalOpenMP(const PolymlpEval& polymlp);
    ~PolymlpEvalOpenMP();

    void eval(
        const vector1i& types,
        const vector2i& neighbor_half,
        const vector3d& neighbor_diff,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );
};

#endif
