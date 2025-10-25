/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params.h"

ModelParams::ModelParams(){}
ModelParams::ModelParams(const feature_params& fp, Maps& maps){

    if (fp.feature_type == "gtinv"){
        uniq_gtinv_type(fp, maps, tp_combs, linear_terms);
    }
    modelp_poly = ModelParamsPoly(fp, maps, linear_terms);
}

ModelParams::~ModelParams(){}

const std::vector<LinearTerm>& ModelParams::get_linear_terms() const {
    return linear_terms;
}
const vector3i& ModelParams::get_tp_combs() const {
    return tp_combs;
}
const int ModelParams::get_n_linear_features() const {
    return modelp_poly.get_n_linear_features();
}

const vector2i& ModelParams::get_comb2() const {
    return modelp_poly.get_comb2();
}
const vector2i& ModelParams::get_comb3() const {
    return modelp_poly.get_comb3();
}
const vector1i& ModelParams::get_comb1_indices(const int type) const {
    return modelp_poly.get_comb1_indices(type);
}
const vector1i& ModelParams::get_comb2_indices(const int type) const {
    return modelp_poly.get_comb2_indices(type);
}
const vector1i& ModelParams::get_comb3_indices(const int type) const {
    return modelp_poly.get_comb3_indices(type);
}
