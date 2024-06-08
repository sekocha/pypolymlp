/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_features_attr.h"

PyFeaturesAttr::PyFeaturesAttr(const py::dict& params_dict){

    const int n_type = params_dict["n_type"].cast<int>();
    const bool& element_swap = params_dict["element_swap"].cast<bool>();

    const py::dict& model = params_dict["model"].cast<py::dict>();
    const auto& pair_params = model["pair_params"].cast<vector2d>();
    const double& cutoff = model["cutoff"].cast<double>();
    const std::string& pair_type = model["pair_type"].cast<std::string>();
    const std::string& feature_type = model["feature_type"].cast<std::string>();
    const int& model_type = model["model_type"].cast<int>();
    const int& maxp = model["max_p"].cast<int>();
    const int& maxl = model["max_l"].cast<int>();

    const py::dict& gtinv = model["gtinv"].cast<py::dict>();
    const auto& lm_array = gtinv["lm_seq"].cast<vector3i>();
    const auto& l_comb = gtinv["l_comb"].cast<vector2i>();
    const auto& lm_coeffs = gtinv["lm_coeffs"].cast<vector2d>();

    const bool force = false;
    struct feature_params fp = {n_type,
                                force,
                                pair_params,
                                cutoff,
                                pair_type,
                                feature_type,
                                model_type,
                                maxp,
                                maxl,
                                lm_array,
                                l_comb,
                                lm_coeffs};

    ModelParams modelp(fp, element_swap);
    type_comb_pair = modelp.get_type_comb_pair();
    const int n_fn = pair_params.size();
    const int n_tc = type_comb_pair.size();

    if (feature_type == "pair"){
        for (int tc = 0; tc < n_tc; ++tc){
            for (int n = 0; n < n_fn; ++n){
                radial_ids.emplace_back(n);
                tcomb_ids.emplace_back(vector1i({tc}));
            }
        }
    }
    else if (feature_type == "gtinv"){
        const auto& linear_all = modelp.get_linear_term_gtinv();
        for (int n = 0; n < n_fn; ++n){
            for (auto& linear: linear_all){
                radial_ids.emplace_back(n);
                gtinv_ids.emplace_back(linear.lmindex);
                tcomb_ids.emplace_back(linear.tcomb_index);
            }
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
const vector3i& PyFeaturesAttr::get_type_comb_pair() const{
    return type_comb_pair;
}
