/****************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

	    Main program for calculating descriptors using Python

****************************************************************************/

#include "model_py.h"

ModelPCAPy::ModelPCAPy(const vector3d& axis, 
                       const vector3d& positions_c,
                       const vector2i& types, 
                       const int& n_type,
                       const bool& force,
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
                       const std::vector<bool>& force_dataset,
                       const vector1i& n_atoms_all,
                       const vector2d& pca_mat,
                       const bool& print_memory){

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

    std::vector<bool> force_st;
    vector1i xf_begin, xs_begin;
    int n_data;
    set_index(n_st_dataset, 
              force_dataset, 
              n_atoms_all, 
              xf_begin, 
              xs_begin, 
              force_st, 
              n_data);

    const int n_pca_features = pca_mat.size();
    const int n_features = pca_mat[0].size();
    const int n_st = axis.size();

    if (print_memory == true){
        std::cout << " matrix shape (X,PCA) = (" 
            << n_data << "," << n_pca_features << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << " Estimated memory allocation = " 
            << double(n_data) * double(n_pca_features) * 8e-9 
            << " (GB)" << std::endl;
        std::cout << std::fixed << std::setprecision(10);
    }

    x_all = Eigen::MatrixXd(n_data, n_pca_features);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1)
    #endif
    for (int i = 0; i < n_st; ++i){
        struct feature_params fp1 = fp;
        fp1.force = force_st[i];

        Neighbor neigh(axis[i], 
                       positions_c[i], 
                       types[i], 
                       fp1.n_type, 
                       fp1.cutoff);
        Model mod(neigh.get_dis_array(), 
                  neigh.get_diff_array(),
                  neigh.get_atom2_array(), 
                  types[i], 
                  fp1, 
                  false);

        const auto &xe = mod.get_xe_sum();
        for (int j = 0; j < n_pca_features; ++j) {
            double val = 0.0;
            for (int k = 0; k < n_features; ++k) val += pca_mat[j][k] * xe[k];
            x_all(i,j) = val;
        }

        if (force_st[i] == true){
            const auto &xf = mod.get_xf_sum();
            for (int j = 0; j < xf.size(); ++j) {
                const int begin1 = xf_begin[i]+j;
                for (int k = 0; k < n_pca_features; ++k) {
                    double val = 0.0;
                    for (int l = 0; l < n_features; ++l) {
                        val += pca_mat[k][l] * xf[j][l];
                    }
                    x_all(begin1,k) = val;
                }
            }
            const auto &xs = mod.get_xs_sum();
            for (int j = 0; j < xs.size(); ++j) {
                const int begin1 = xs_begin[i]+j;
                for (int k = 0; k < n_pca_features; ++k) {
                    double val = 0.0;
                    for (int l = 0; l < n_features; ++l) {
                        val += pca_mat[k][l] * xs[j][l];
                    }
                    x_all(begin1,k) = val;
                }
            }
        }
    }
}

ModelPCAPy::~ModelPCAPy(){}

void ModelPCAPy::set_index(const std::vector<int>& n_data_dataset, 
                           const std::vector<bool>& force_dataset,
                           const std::vector<int>& n_atoms_st,
                           std::vector<int>& xf_begin, 
                           std::vector<int>& xs_begin,
                           std::vector<bool>& force, 
                           int& n_row){

    int n_st = std::accumulate(n_data_dataset.begin(),n_data_dataset.end(),0);

    xs_begin_dataset = xf_begin_dataset = vector1i(n_data_dataset.size(), -1);
    xs_begin = xf_begin = vector1i(n_st, -1);
    force = std::vector<bool>(n_st, false);

    int iforce = n_st, istress = n_st;
    for (int i = 0; i < n_data_dataset.size(); ++i){
        if (force_dataset[i] == true) iforce += n_data_dataset[i] * 6;
    }

    int n = 0; 
    n_row = n_st;
    for (int i = 0; i < n_data_dataset.size(); ++i){
        if (force_dataset[i] == true){
            xf_begin_dataset[i] = iforce;
            xs_begin_dataset[i] = istress;
        }
        for (int j = 0; j < n_data_dataset[i]; ++j){
            if (force_dataset[i] == true){
                xf_begin[n] = iforce;
                xs_begin[n] = istress;
                force[n] = true;
                iforce += 3 * n_atoms_st[n];
                istress += 6;
                n_row += 6 + 3 * n_atoms_st[n];
            }
            ++n;
        }
    }
}

Eigen::MatrixXd& ModelPCAPy::get_x(){ return x_all; }
const vector1i& ModelPCAPy::get_fbegin() const{ return xf_begin_dataset; }
const vector1i& ModelPCAPy::get_sbegin() const{ return xs_begin_dataset; }

