/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_EVAL
#define __POLYMLP_EVAL

#include "mlpcpp.h"

#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_model_params.h"
#include "polymlp/polymlp_features.h"
#include "polymlp/polymlp_potential.h"

class PolymlpEval {

    struct DataPolyMLP {
        struct feature_params fp;
        ModelParams modelp;
        Potential p_obj;
    };

    struct DataPolyMLP pot;
    vector2i type_comb;
    void set_type_comb();

    /* for feature_type = pair */
    void compute_antc(const vector2d& positions_c,
                      const vector1i& types,
                      const vector2i& neighbor_half,
                      const vector3d& neighbor_diff,
                      vector2d& antc);

    void compute_sum_of_prod_antc(const vector1i& types,
                                  const vector2d& antc,
                                  vector2d& prod_antc_sum_e,
                                  vector2d& prod_antc_sum_f);

    void eval_pair(const vector2d& positions_c,
                   const vector1i& types,
                   const vector2i& neighbor_half,
                   const vector3d& neighbor_diff,
                   double& energy,
                   vector2d& forces,
                   vector1d& stress);

    /* for feature_type = gtinv */
    void compute_anlmtc(const vector2d& positions_c,
                        const vector1i& types,
                        const vector2i& neighbor_half,
                        const vector3d& neighbor_diff,
                        vector2dc& anlmtc);
    void compute_anlmtc_conjugate(const vector2d& anlmtc_r,
                                  const vector2d& anlmtc_i,
                                  vector2dc& anlmtc);
    void compute_sum_of_prod_anlmtc(const vector1i& types,
                                    const vector2dc& anlmtc,
                                    vector2dc& prod_sum_e,
                                    vector2dc& prod_sum_f);

    void compute_linear_features(const vector1d& prod_anlmtc,
                                 const int type1,
                                 vector1d& feature_values);
    template<typename T>
    void compute_products(const vector2i& map,
                          const std::vector<T>& element,
                          std::vector<T>& prod_vals);

    void compute_products_real(const vector2i& map,
                               const vector1dc& element,
                               vector1d& prod_vals);

    double prod_real(const dc& val1, const dc& val2);
    dc prod_real_and_complex(const double val1, const dc& val2);

    void eval_gtinv(const vector2d& positions_c,
                    const vector1i& types,
                    const vector2i& neighbor_half,
                    const vector3d& neighbor_diff,
                    double& energy,
                    vector2d& forces,
                    vector1d& stress);

    public:

    PolymlpEval();
    PolymlpEval(const feature_params& fp, const vector1d& coeffs);
    ~PolymlpEval();

    void eval(const vector2d& positions_c,
              const vector1i& types,
              const vector2i& neighbor_half,
              const vector3d& neighbor_diff,
              double& energy,
              vector2d& forces,
              vector1d& stress);
};

#endif
