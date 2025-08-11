/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MODEL_PARAMS
#define __POLYMLP_MODEL_PARAMS

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_model_params_gtinv.h"
#include "polymlp_model_params_polynomial.h"


class ModelParams{

    int n_type;
    std::vector<LinearTerm> linear_terms;
    vector3i tp_combs;
    ModelParamsPoly modelp_poly;

    public:

    ModelParams();
    ModelParams(const feature_params& fp, Maps& maps);
    ~ModelParams();

    const std::vector<LinearTerm>& get_linear_terms() const;
    const vector3i& get_tp_combs() const;
    const int get_n_linear_features() const;

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;
    const vector1i& get_comb1_indices(const int type) const;
    const vector1i& get_comb2_indices(const int type) const;
    const vector1i& get_comb3_indices(const int type) const;

};

#endif
