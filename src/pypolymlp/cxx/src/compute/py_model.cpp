/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_model.h"

PyModel::PyModel(const py::dict& params_dict,
                 const vector3d& axis,
                 const vector3d& positions_c,
                 const vector2i& types,
                 const vector1i& n_st_dataset,
                 const std::vector<bool>& force_dataset,
                 const vector1i& n_atoms_all){

    struct feature_params fp;
    const bool& element_swap = params_dict["element_swap"].cast<bool>();
    const bool& print_memory = params_dict["print_memory"].cast<bool>();
    convert_params_dict_to_feature_params(params_dict, fp);

    const Features f_obj(fp);
    const FunctionFeatures features_obj(f_obj);

    std::vector<bool> force_st;
    vector1i xf_begin, xs_begin;
    set_index(
        n_st_dataset, force_dataset, n_atoms_all,
        xf_begin, xs_begin, force_st
    );

    Neighbor neigh(axis[0], positions_c[0], types[0], fp.n_type, fp.cutoff);
    ModelFast mod(
        neigh.get_dis_array(), neigh.get_diff_array(), neigh.get_atom2_array(),
        types[0], fp, features_obj
    );

    const int n_features = mod.get_xe_sum().size();
    const int n_st = axis.size();
    const int total_n_data = n_data[0] + n_data[1] + n_data[2];

    if (print_memory == true){
        std::cout << " matrix shape (X): ("
            << total_n_data << "," << n_features << ")" << std::endl;
    }

    x_all = Eigen::MatrixXd(total_n_data, n_features);
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
        ModelFast mod(neigh.get_dis_array(),
                      neigh.get_diff_array(),
                      neigh.get_atom2_array(),
                      types[i], fp1, features_obj);

        const auto &xe = mod.get_xe_sum();
        for (size_t j = 0; j < xe.size(); ++j) x_all(i,j) = xe[j];

        if (force_st[i] == true){
            const auto &xf = mod.get_xf_sum();
            const auto &xs = mod.get_xs_sum();
            for (size_t j = 0; j < xf.size(); ++j) {
                for (size_t k = 0; k < xf[j].size(); ++k){
                    x_all(xf_begin[i]+j, k) = xf[j][k];
                }
            }
            for (size_t j = 0; j < xs.size(); ++j) {
                for (size_t k = 0; k < xs[j].size(); ++k){
                    x_all(xs_begin[i]+j, k) = xs[j][k];
                }
            }
        }
    }
}

PyModel::~PyModel(){}

void PyModel::set_index(const std::vector<int>& n_data_dataset,
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

Eigen::MatrixXd& PyModel::get_x(){ return x_all; }
const vector1i& PyModel::get_fbegin() const{ return xf_begin_dataset; }
const vector1i& PyModel::get_sbegin() const{ return xs_begin_dataset; }
const vector1i& PyModel::get_n_data() const{ return n_data; }
