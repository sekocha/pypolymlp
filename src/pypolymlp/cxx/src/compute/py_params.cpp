/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

 ******************************************************************************/

#include "py_params.h"


void convert_params_dict_to_feature_params(const py::dict& params_dict,
                                           feature_params& fp){

    const int n_type = params_dict["n_type"].cast<int>();
    const bool& element_swap = params_dict["element_swap"].cast<bool>();
    const bool& print_memory = params_dict["print_memory"].cast<bool>();

    const py::dict& model = params_dict["model"].cast<py::dict>();
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

    const bool& pair_cond = model["pair_conditional"].cast<bool>();
    const auto& pair_params = model["pair_params"].cast<vector2d>();

    vector3i pair_params_cond;
    const auto& dict1 = model["pair_params_conditional"].cast<py::dict>();
    pair_params_cond.resize(n_type);
    for (int i = 0; i < n_type; ++i){
        pair_params_cond[i].resize(n_type);
        for (int j = 0; j <= i; ++j){
            py::tuple tup = py::make_tuple(j, i);
            const auto& params = dict1[tup].cast<vector1i>();
            pair_params_cond[j][i] = params;
        }
    }

    const bool force = false;
    fp = {n_type,
          force,
          pair_params,
          pair_params_cond,
          cutoff,
          pair_type,
          feature_type,
          model_type,
          maxp,
          maxl,
          lm_array,
          l_comb,
          lm_coeffs};

}
