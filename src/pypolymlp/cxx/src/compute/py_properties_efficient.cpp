/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_properties_efficient.h"

PyPropertiesFast::PyPropertiesFast(const py::dict& params_dict,
                                   const vector1d& coeffs,
                                   const vector3d& axis, 
                                   const vector3d& positions_c,
                                   const vector2i& types){

    const int n_type = params_dict["n_type"].cast<int>();
    //const bool& element_swap = params_dict["element_swap"].cast<bool>();

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

    const bool force = true;
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

    const int n_st = axis.size();
    e_all = Eigen::VectorXd(n_st);
    f_all = vector3d(n_st);
    s_all = Eigen::MatrixXd(n_st, 6);

    PolymlpEval polymlp(fp, coeffs);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1)
    #endif
    for (int i = 0; i < n_st; ++i){
        NeighborHalf neigh(axis[i], positions_c[i], types[i], fp.cutoff);
        vector1d stress;
        polymlp.eval(positions_c[i],
                     types[i],
                     neigh.get_half_list(),
                     neigh.get_diff_list(),
                     e_all(i), 
                     f_all[i],
                     stress);
        for (int j = 0; j < 6; ++j) s_all(i,j) = stress[j];
    }
}

PyPropertiesFast::~PyPropertiesFast(){}

Eigen::VectorXd& PyPropertiesFast::get_e(){ return e_all; }
const vector3d& PyPropertiesFast::get_f() const { return f_all; }
Eigen::MatrixXd& PyPropertiesFast::get_s(){ return s_all; }


