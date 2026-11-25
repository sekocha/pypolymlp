/****************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

	    Main program for calculating descriptors using Python

****************************************************************************/

#include "model_charge_py.h"

ModelForChargePy::ModelForChargePy(const vector3d& axis,
                                   const vector3d& positions_c,
                                   const vector2i& types,
                                   const int& n_type,
                                   const vector2d& params,
                                   const double& cutoff,
                                   const std::string& pair_type,
                                   const std::string& des_type,
                                   const int& model_type,
                                   const int& maxp,
                                   const int& maxl,
                                   const vector3i& lm_array,
                                   const vector2i& l_comb,
                                   const vector2d& lm_coeffs,
                                   const vector1i& n_st_dataset,
                                   const bool& print_memory){

    bool force = false;
    struct feature_params fp = {n_type,
                                force,
                                params,
                                cutoff,
                                pair_type,
                                des_type,
                                model_type,
                                maxp,
                                maxl,
                                lm_array,
                                l_comb,
                                lm_coeffs};

    vector1i xc_size_array, xc_begin_idx_array;
    int n_data(0);
    for (const auto& t1: types){
        int n_count = t1.size();
        xc_size_array.emplace_back(n_count);
        xc_begin_idx_array.emplace_back(n_data);
        n_data += n_count;
    }
    const int n_st = xc_size_array.size();

    xc_begin_idx_dataset = vector1i(n_st_dataset.size(), -1);

    int n = 0;
    int idx = 0;
    for (int i = 0; i < n_st_dataset.size(); ++i){
        xc_begin_idx_dataset[i] = idx;
        for (int j = 0; j < n_st_dataset[i]; ++j){
            idx += xc_size_array[n];
            ++n;
        }
    }

    Neighbor neigh(axis[0], positions_c[0], types[0], fp.n_type, fp.cutoff);
    ComputeFeatures mod(neigh.get_dis_array(),
                        neigh.get_diff_array(),
                        neigh.get_atom2_array(),
                        types[0],
                        fp);
    const int n_terms = mod.get_x()[0].size();

    if (print_memory == true){
        std::cout << " matrix shape (X, charge) = ("
                  << n_data << "," << n_terms << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << " Estimated memory allocation = "
                  << double(n_data) * double(n_terms) * 8e-9
                  << " (GB)" << std::endl;
        std::cout << std::fixed << std::setprecision(10);
    }

    x_all = Eigen::MatrixXd(n_data, n_terms);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1)
    #endif
    for (int i = 0; i < n_st; ++i){
        struct feature_params fp1 = fp;
        Neighbor neigh(axis[i],
                       positions_c[i],
                       types[i],
                       fp1.n_type,
                       fp1.cutoff);
        ComputeFeatures mod(neigh.get_dis_array(),
                            neigh.get_diff_array(),
                            neigh.get_atom2_array(),
                            types[i],
                            fp);

        const auto &xc = mod.get_x();
        for (int j = 0; j < xc.size(); ++j) {
            for (int k = 0; k < xc[j].size(); ++k){
                x_all(xc_begin_idx_array[i]+j, k) = xc[j][k];
            }
        }
    }
}

ModelForChargePy::~ModelForChargePy(){}

Eigen::MatrixXd& ModelForChargePy::get_x(){
    return x_all;
}
const vector1i& ModelForChargePy::get_cbegin() const {
    return xc_begin_idx_dataset;
}
