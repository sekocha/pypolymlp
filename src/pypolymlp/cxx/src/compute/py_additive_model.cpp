/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

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
    std::vector<ModelParams> modelp_array;
    std::vector<FunctionFeatures> features_array;
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

        // added for local_fast and model_fast
        ModelParams modelp(fp, element_swap);
        modelp_array.emplace_back(modelp);

        const Features f_obj(fp, modelp);
        FunctionFeatures features_obj(fp, modelp, f_obj);
        features_array.emplace_back(features_obj);
        //
    }

    std::vector<bool> force_st;
    vector1i xf_begin, xs_begin;
    set_index(n_st_dataset,
              force_dataset,
              n_atoms_all,
              xf_begin,
              xs_begin,
              force_st);

    const int n_st = axis.size();
    const int total_n_data = n_data[0] + n_data[1] + n_data[2];
    int n_features(0), imodel(0);
    for (const auto& fp: fp_array){

        std::set<int> uniq_types(types[0].begin(), types[0].end());

        vector1i types_mod = modify_types(types[0],
                                          uniq_types.size(),
                                          fp.n_type);

        Neighbor neigh(axis[0], positions_c[0], types_mod,
                       fp.n_type, fp.cutoff);
        ModelFast mod(neigh.get_dis_array(),
                      neigh.get_diff_array(),
                      neigh.get_atom2_array(),
                      types_mod, fp,
                      modelp_array[imodel], features_array[imodel]);

        n_features += mod.get_xe_sum().size();
        cumulative_n_features.emplace_back(n_features);
        ++imodel;
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
        std::set<int> uniq_types(types[i].begin(), types[i].end());
        for (size_t n = 0; n < cumulative_n_features.size(); ++n){
            struct feature_params fp1 = fp_array[n];
            const auto& modelp1 = modelp_array[n];
            const auto& features1 = features_array[n];
            fp1.force = force_st[i];

            int first_index;
            if (n == 0) first_index = 0;
            else first_index = cumulative_n_features[n-1];

            vector1i types_mod = modify_types(types[i],
                                              uniq_types.size(),
                                              fp1.n_type);
            Neighbor neigh(axis[i],
                           positions_c[i],
                           types_mod,
                           fp1.n_type,
                           fp1.cutoff);
            ModelFast mod(neigh.get_dis_array(),
                          neigh.get_diff_array(),
                          neigh.get_atom2_array(),
                          types_mod, fp1,
                          modelp1, features1);

            const auto &xe = mod.get_xe_sum();
            for (size_t j = 0; j < xe.size(); ++j)
                x_all(i,first_index+j) = xe[j];

            if (force_st[i] == true){
                const auto &xf = mod.get_xf_sum();
                const auto &xs = mod.get_xs_sum();
                for (size_t j = 0; j < xf.size(); ++j) {
                    for (size_t k = 0; k < xf[j].size(); ++k){
                        x_all(xf_begin[i]+j, first_index+k) = xf[j][k];
                    }
                }
                for (size_t j = 0; j < xs.size(); ++j) {
                    for (size_t k = 0; k < xs[j].size(); ++k){
                        x_all(xs_begin[i]+j, first_index+k) = xs[j][k];
                    }
                }
            }
        }
    }
}

PyAdditiveModel::~PyAdditiveModel(){}

vector1i PyAdditiveModel::modify_types(const std::vector<int>& types_orig,
                                       const int n_type_orig,
                                       const int n_type){
    vector1i types_mod;
    if (n_type == 1 and n_type != n_type_orig){
        types_mod = vector1i(types_orig.size(), 0);
    }
    else types_mod = types_orig;
    return types_mod;
}

void PyAdditiveModel::set_index(const std::vector<int>& n_data_dataset,
                                const std::vector<bool>& force_dataset,
                                const std::vector<int>& n_atoms_st,
                                std::vector<int>& xf_begin,
                                std::vector<int>& xs_begin,
                                std::vector<bool>& force){

    const int n_st = std::accumulate(n_data_dataset.begin(),
                                     n_data_dataset.end(), 0);
    const int n_datasets = n_data_dataset.size();

    xs_begin_dataset = xf_begin_dataset = vector1i(n_datasets, -1);
    xs_begin = xf_begin = vector1i(n_st, -1);
    force = std::vector<bool>(n_st, false);
    n_data = vector1i(3, 0);
    n_data[0] = n_st;

    int id_st(0), istress(n_st);
    for (int i = 0; i < n_datasets; ++i){
        if (force_dataset[i] == true) {
            xs_begin_dataset[i] = istress;
            n_data[2] += n_data_dataset[i] * 6;
        }
        for (int j = 0; j < n_data_dataset[i]; ++j){
            if (force_dataset[i] == true){
                xs_begin[id_st] = istress;
                istress += 6;
            }
            ++id_st;
        }
    }

    id_st = 0;
    int iforce = istress;
    for (int i = 0; i < n_datasets; ++i){
        if (force_dataset[i] == true) {
            xf_begin_dataset[i] = iforce;
        }
        for (int j = 0; j < n_data_dataset[i]; ++j){
            if (force_dataset[i] == true){
                force[id_st] = true;
                xf_begin[id_st] = iforce;
                iforce += 3 * n_atoms_st[id_st];
                n_data[1] += 3 * n_atoms_st[id_st];
            }
            ++id_st;
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
