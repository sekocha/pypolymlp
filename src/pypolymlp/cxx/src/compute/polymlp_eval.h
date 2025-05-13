/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_EVAL
#define __POLYMLP_EVAL

#include "mlpcpp.h"

#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_products.h"

#include "polymlp/polymlp_mapping.h"
#include "polymlp/polymlp_model_params.h"
#include "polymlp/polymlp_features.h"
#include "polymlp/polymlp_potential.h"


struct DataPolyMLP {
    struct feature_params fp;
    Mapping mapping;
    ModelParams modelp;
    Potential p_obj;
};


class PolymlpEval {

    vector2i type_pairs;

    /* for feature_type = pair */
    void compute_antp(const vector1i& types,
                      const vector2i& neighbor_half,
                      const vector3d& neighbor_diff,
                      vector2d& antp);

    void compute_sum_of_prod_antp(const vector1i& types,
                                  const vector2d& antp,
                                  vector2d& prod_antp_sum_e,
                                  vector2d& prod_antp_sum_f);

    void eval_pair(const vector1i& types,
                   const vector2i& neighbor_half,
                   const vector3d& neighbor_diff,
                   double& energy,
                   vector2d& forces,
                   vector1d& stress);

    /* for feature_type = gtinv */
    void compute_anlmtp(const vector1i& types,
                        const vector2i& neighbor_half,
                        const vector3d& neighbor_diff,
                        vector2dc& anlmtp);
    void compute_anlmtp_conjugate(const vector2d& anlmtp_r,
                                  const vector2d& anlmtp_i,
                                  vector2dc& anlmtp);

    void compute_sum_of_prod_anlmtp(const vector1i& types,
                                    const vector2dc& anlmtp,
                                    vector2dc& prod_sum_e,
                                    vector2dc& prod_sum_f);

    void compute_linear_features(const vector1d& prod_anlmtp,
                                 const int type1,
                                 vector1d& feature_values);


    void eval_gtinv(const vector1i& types,
                    const vector2i& neighbor_half,
                    const vector3d& neighbor_diff,
                    double& energy,
                    vector2d& forces,
                    vector1d& stress);

    public:

    PolymlpEval();
    PolymlpEval(const feature_params& fp, const vector1d& coeffs);
    ~PolymlpEval();

    struct DataPolyMLP pot;
    void eval(const vector1i& types,
              const vector2i& neighbor_half,
              const vector3d& neighbor_diff,
              double& energy,
              vector2d& forces,
              vector1d& stress);
};

#endif
