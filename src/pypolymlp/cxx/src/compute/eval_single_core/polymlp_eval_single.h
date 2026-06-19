/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_EVAL_SINGLE
#define __POLYMLP_EVAL_SINGLE

#include "mlpcpp.h"
#include "polymlp/polymlp_api.h"
#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_products.h"
#include "compute/neighbor_half_single.h"


class PolymlpEvalSingle {

    PolymlpAPI polymlp_api;
    int n_atom;
    std::vector<std::vector<std::vector<nlmtpAttr> > > nlmtp_attrs;

    void convert_neighbor_half_to_full(
        NeighborHalfSingle& neigh,
        vector1i& neighbor_full,
        vector1d& dx_full,
        vector1d& dy_full,
        vector1d& dz_full,
        vector1i& offset);

    /* for feature_type = pair */
    void compute_antp(
        const vector1i& types,
        NeighborHalfSingle& neigh,
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
        NeighborHalfSingle& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    /* for feature_type = gtinv */
    void compute_sum_of_prod_anlmtp(
        const vector1i& types,
        NeighborHalfSingle& neigh,
        vector2dc& prod_sum_e,
        vector2dc& prod_sum_f);

    void eval_gtinv(
        const vector1i& types,
        NeighborHalfSingle& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    void collect_properties(
        const vector2d& e_array,
        const vector2d& fx_array,
        const vector2d& fy_array,
        const vector2d& fz_array,
        NeighborHalfSingle& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );

    public:

    PolymlpEvalSingle();
    PolymlpEvalSingle(const feature_params& fp, const vector1d& coeffs);
    ~PolymlpEvalSingle();

    void eval(
        const vector1i& types,
        NeighborHalfSingle& neigh,
        double& energy,
        vector2d& forces,
        vector1d& stress
    );
};

#endif
