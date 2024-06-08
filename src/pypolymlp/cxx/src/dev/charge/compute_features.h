/****************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __COMPUTE_FEATURES
#define __COMPUTE_FEATURES

#include "mlpcpp.h"

#include "polymlp/polymlp_model_params.h"
#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_features.h"
#include "polymlp/polymlp_potential.h"

using vector1map_d = std::vector<std::map<int, double> >;
using vector2map_d = std::vector<vector1map_d>;
using vector1map_dc = std::vector<std::map<int, dc> >;
using vector2map_dc = std::vector<vector1map_dc>;

class ComputeFeatures {

    int n_atom, n_type, n_terms;
    ModelParams modelp;
    Potential p_obj;

    vector1i types;
    vector2d xc;

    // for pair --------------------------------------------------
    void compute_antc(const vector3d& dis_array,
                      const feature_params& fp,
                      vector2d& antc);

    void compute_sum_of_prod_antc(const vector2d& antc,
                                  vector2map_d& prod_sum_e);

    void compute_features_charge(const vector3d& dis_array,
                                 const vector3i& atom2_array,
                                 const feature_params& fp,
                                 const vector2map_d& prod_sum_e);
    // -----------------------------------------------------------

    // for gtinv -------------------------------------------------
    void compute_anlmtc(const vector3d& dis_array,
                        const vector4d& diff_array,
                        const feature_params& fp,
                        vector2dc& anlmtc);
    void compute_anlmtc_conjugate(const vector2d& anlmtc_r,
                                  const vector2d& anlmtc_i,
                                  vector2dc& anlmtc);
    void compute_sum_of_prod_anlmtc(const vector2dc& anlmtc,
                                    vector2map_dc& prod_sum_e);
    void compute_linear_features(const vector1dc& prod_anlmtc,
                                 vector1d& feature_values);

    void compute_features_charge(const vector3d& dis_array,
                                 const vector4d& diff_array,
                                 const vector3i& atom2_array,
                                 const feature_params& fp,
                                 const vector2map_dc& prod_sum_e);

    double prod_real(const dc& val1, const dc& val2);
    // -----------------------------------------------------------

    // for both pair and gtinv
    template<typename T>
    void compute_products(const vector2i& map,
                          const std::vector<T>& element,
                          std::vector<T>& prod_vals);

    public:

    ComputeFeatures();
    ComputeFeatures(const vector3d& dis_array_all,
                    const vector4d& diff_array_all,
                    const vector3i& atom2_array,
                    const vector1i& types_i,
                    const struct feature_params& fp);
    ~ComputeFeatures();

    const vector2d& get_x() const;

};

#endif
