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
#include "compute/neighbor_half_openmp.h"

struct Diff {
    double x, y, z;
};


class PolymlpEvalOpenMP {

    PolymlpAPI polymlp_api;
    int n_atom;

    void convert_neighbor_half_to_full(
        NeighborHalfOpenMP& neigh,
        vector1i& neighbor_full,
        std::vector<Diff>& neighbor_diff_full,
        vector1i& offset);


    /* for feature_type = pair */
    void compute_antp(
        const vector1i& types,
        NeighborHalfOpenMP& neigh,
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
        NeighborHalfOpenMP& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    /* for feature_type = gtinv */
    void compute_anlmtp(
        const vector1i& types,
        NeighborHalfOpenMP& neigh,
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
        NeighborHalfOpenMP& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    void collect_properties(
        const vector2d& e_array,
        const vector2d& fx_array,
        const vector2d& fy_array,
        const vector2d& fz_array,
        NeighborHalfOpenMP& neigh,
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
        NeighborHalfOpenMP& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );
};

#endif
