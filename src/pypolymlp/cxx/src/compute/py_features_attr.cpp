/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_features_attr.h"

PyFeaturesAttr::PyFeaturesAttr(const py::dict& params_dict){

    struct feature_params fp;
    const bool& element_swap = params_dict["element_swap"].cast<bool>();
    convert_params_dict_to_feature_params(params_dict, fp);

    Mapping mapping(fp);
    ModelParams modelp(fp, mapping);
    type_pairs = mapping.get_type_pairs();
    // TODO: must be revised.
    const int n_fn = fp.params.size();
    const int n_tp = mapping.get_n_type_pairs();

    if (fp.feature_type == "pair"){
        for (int tp = 0; tp < n_tp; ++tp){
            for (int n = 0; n < n_fn; ++n){
                radial_ids.emplace_back(n);
                tcomb_ids.emplace_back(vector1i({tp}));
            }
        }
    }
    /*
    else if (fp.des_type == "gtinv"){
        const auto& linear_all = modelp.get_linear_term_gtinv();
        for (int n = 0; n < n_fn; ++n){
            for (auto& linear: linear_all){
                radial_ids.emplace_back(n);
                gtinv_ids.emplace_back(linear.lmindex);
                tcomb_ids.emplace_back(linear.tcomb_index);
            }
        }
    }
    */

    for (const auto& comb2: modelp.get_comb2()){
        polynomial_ids.emplace_back(comb2);
    }
    for (const auto& comb3: modelp.get_comb3()){
        polynomial_ids.emplace_back(comb3);
    }
}

PyFeaturesAttr::~PyFeaturesAttr(){}

const vector1i& PyFeaturesAttr::get_radial_ids() const{
    return radial_ids;
}
const vector1i& PyFeaturesAttr::get_gtinv_ids() const{
    return gtinv_ids;
}
const vector2i& PyFeaturesAttr::get_tcomb_ids() const{
    return tcomb_ids;
}
const vector2i& PyFeaturesAttr::get_polynomial_ids() const{
    return polynomial_ids;
}
const vector2i& PyFeaturesAttr::get_type_pairs() const{
    return type_pairs;
}
