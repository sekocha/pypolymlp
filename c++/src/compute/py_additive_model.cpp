/****************************************************************************

        Copyright (C) 2023 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to
        the Free Software Foundation, Inc., 51 Franklin Street,
        Fifth Floor, Boston, MA 02110-1301, USA, or see
        http://www.gnu.org/copyleft/gpl.txt

	    Main program for calculating descriptors using Python

****************************************************************************/

#include "py_additive_model.h"

PyAdditiveModel::PyAdditiveModel(const std::vector<py::dict>& params_dict_array,
                                 const vector3d& axis, 
                                 const vector3d& positions_c,
                                 const vector2i& types, 
                                 const vector1i& n_st_dataset,
                                 const std::vector<bool>& force_dataset,
                                 const vector1i& n_atoms_all){

    std::vector<struct feature_params> fp_array;
    bool element_swap, print_memory;
    for (const auto& params_dict: params_dict_array){
        const int n_type = params_dict["n_type"].cast<int>();
        element_swap = params_dict["element_swap"].cast<bool>();
        print_memory = params_dict["print_memory"].cast<bool>();

        const py::dict& model = params_dict["model"].cast<py::dict>();
        const auto& pair_params = model["pair_params"].cast<vector2d>();
        const double& cutoff = model["cutoff"].cast<double>();
        const std::string& pair_type = model["pair_type"].cast<std::string>();
        const std::string& feature_type 
                = model["feature_type"].cast<std::string>();
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
        fp_array.emplace_back(fp);
    }

    std::vector<bool> force_st;
    vector1i xf_begin, xs_begin;
    int total_n_data;
    set_index(n_st_dataset, 
              force_dataset, 
              n_atoms_all, 
              xf_begin, 
              xs_begin, 
              force_st, 
              total_n_data);

    const int n_st = axis.size();
    int n_features(0);
    for (const auto& fp: fp_array){
        Neighbor neigh(axis[0], positions_c[0], types[0], fp.n_type, fp.cutoff);
        Model mod(neigh.get_dis_array(), 
                  neigh.get_diff_array(),
                  neigh.get_atom2_array(), 
                  types[0], 
                  fp, 
                  element_swap);
        n_features += mod.get_xe_sum().size();
        cumulative_n_features.emplace_back(n_features);
    }

    if (print_memory == true){
        std::cout << " matrix shape (X) = (" 
            << total_n_data << "," << n_features << ")" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << " Estimated memory allocation = " 
            << double(total_n_data) * double(n_features) * 8e-9 
            << " (GB)" << std::endl;
        std::cout << std::fixed << std::setprecision(10);
    }

    x_all = Eigen::MatrixXd(total_n_data, n_features);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1)
    #endif
    for (int i = 0; i < n_st; ++i){
        for (int n = 0; n < cumulative_n_features.size(); ++n){
            struct feature_params fp1 = fp_array[n];
            fp1.force = force_st[i];

            int first_index;
            if (n == 0) first_index = 0;
            else first_index = cumulative_n_features[n-1];

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
                      element_swap);

            const auto &xe = mod.get_xe_sum();
            for (int j = 0; j < xe.size(); ++j) x_all(i,first_index+j) = xe[j];

            if (force_st[i] == true){
                const auto &xf = mod.get_xf_sum();
                const auto &xs = mod.get_xs_sum();
                for (int j = 0; j < xf.size(); ++j) {
                    for (int k = 0; k < xf[j].size(); ++k){
                        x_all(xf_begin[i]+j, first_index+k) = xf[j][k];
                    }
                }
                for (int j = 0; j < xs.size(); ++j) {
                    for (int k = 0; k < xs[j].size(); ++k){
                        x_all(xs_begin[i]+j, first_index+k) = xs[j][k];
                    }
                }
            }
        }
    }
}

PyAdditiveModel::~PyAdditiveModel(){}

void PyAdditiveModel::set_index(const std::vector<int>& n_data_dataset, 
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

    n_data = vector1i(3, 0);
    n_data[0] = n_st;

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
                n_data[1] += 3 * n_atoms_st[n];
                n_data[2] += 6;
            }
            ++n;
        }
    }
}

Eigen::MatrixXd& PyAdditiveModel::get_x(){ return x_all; }

const vector1i& PyAdditiveModel::get_fbegin() const{ return xf_begin_dataset; }

const vector1i& PyAdditiveModel::get_sbegin() const{ return xs_begin_dataset; }

const vector1i& PyAdditiveModel::get_cumulative_n_features() const{ 
    return cumulative_n_features; 
}

const vector1i& PyAdditiveModel::get_n_data() const{ return n_data; }
