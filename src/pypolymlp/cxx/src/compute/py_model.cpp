/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_model.h"
#include <chrono>

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

    std::vector<bool> force_st;
    vector1i xf_begin, xs_begin;
    set_index(n_st_dataset, force_dataset, n_atoms_all, xf_begin, xs_begin, force_st);

    Model mod(fp);
    const int n_st = axis.size();
    const int total_n_data = n_data[0] + n_data[1] + n_data[2];
    const int n_features = mod.get_n_features();

    if (print_memory == true){
        std::cout << " Matrix shape (X): ("
            << total_n_data << "," << n_features << ")" << std::endl;
        double mem = double(total_n_data) * double(n_features) * 8 * 1e-9;
        std::cout << " Required memory for X: " << std::setprecision(5)
            << mem << " (GB)" << std::endl;
    }

    x_all = Eigen::MatrixXd(total_n_data, n_features);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1)
    #endif
    for (int i = 0; i < n_st; ++i){
        Neighbor neigh(axis[i], positions_c[i], types[i], fp.n_type, fp.cutoff);
        auto t2 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd xe;
        Eigen::MatrixXd xf, xs;
        mod.run(
            neigh.get_dis_array(),
            neigh.get_diff_array(),
            neigh.get_atom2_array(),
            types[i],
            force_st[i],
            xe, xf, xs
        );
        auto t3 = std::chrono::high_resolution_clock::now();
        x_all.row(i) = xe.transpose();
        if (force_st[i]) {
            x_all.block(xf_begin[i], 0, xf.rows(), xf.cols()) = xf;
            x_all.block(xs_begin[i], 0, xs.rows(), xs.cols()) = xs;
        }
        auto t4 = std::chrono::high_resolution_clock::now();

        //auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
        //auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
        //std::cout << "t2: " << duration2.count() << " ms" << std::endl;
        //std::cout << "t3: " << duration3.count() << " ms" << std::endl;
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
