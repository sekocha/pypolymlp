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

    if (fp.feature_type == "pair"){
        const auto& ntp_attrs = mapping.get_ntp_attrs();
        for (const auto& ntp: ntp_attrs){
            radial_ids.emplace_back(ntp.n);
            tcomb_ids.emplace_back(vector1i({ntp.tp}));
        }
    }
    else if (fp.feature_type == "gtinv"){
        const auto& linear_terms = modelp.get_linear_terms();
        const auto& tp_combs = modelp.get_tp_combs();
        for (const auto& linear: linear_terms){
            radial_ids.emplace_back(linear.n);
            gtinv_ids.emplace_back(linear.lm_comb_id);
            tcomb_ids.emplace_back(tp_combs[linear.order][linear.tp_comb_id]);
        }
    }

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
