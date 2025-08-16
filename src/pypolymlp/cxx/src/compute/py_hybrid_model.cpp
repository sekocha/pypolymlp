/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_hybrid_model.h"

PyHybridModel::PyHybridModel(
    const std::vector<py::dict>& params_dict_array,
    const vector3d& axis,
    const vector3d& positions_c,
    const vector2i& types,
    const vector1i& n_st_dataset,
    const std::vector<bool>& force_dataset,
    const vector1i& n_atoms_all
){

    std::vector<feature_params> fp_array;
    vector2i type_indices_array;
    std::vector<bool> type_full_array;
    bool element_swap, print_memory;
    for (const auto& params_dict: params_dict_array){
        feature_params fp;
        element_swap = params_dict["element_swap"].cast<bool>();
        print_memory = params_dict["print_memory"].cast<bool>();
        convert_params_dict_to_feature_params(params_dict, fp);
        fp_array.emplace_back(fp);
        type_indices_array.emplace_back(params_dict["type_indices"].cast<vector1i>());
        type_full_array.emplace_back(params_dict["type_full"].cast<bool>());
    }

    std::vector<bool> force_st;
    vector1i xf_begin, xs_begin;
    set_index(
        n_st_dataset,
        force_dataset,
        n_atoms_all,
        xf_begin,
        xs_begin,
        force_st
    );

    const int n_st = axis.size();
    const int total_n_data = n_data[0] + n_data[1] + n_data[2];
    int n_features(0), imodel(0);
    std::vector<Model> model_array;
    for (const auto& fp: fp_array){
        vector1i active_atoms, types_active;
        vector2d positions_c_active;
        find_active_atoms(
            type_full_array[imodel],
            type_indices_array[imodel],
            types[0],
            positions_c[0],
            active_atoms,
            types_active,
            positions_c_active
        );
        Neighbor neigh(axis[0], positions_c_active, types_active, fp.n_type, fp.cutoff);

        vector1d xe;
        vector2d xf, xs;
        Model mod(fp);
        model_array.emplace_back(mod);
        mod.run(
            neigh.get_dis_array(),
            neigh.get_diff_array(),
            neigh.get_atom2_array(),
            types_active,
            false,
            xe, xf, xs
        );

        n_features += xe.size();
        cumulative_n_features.emplace_back(n_features);
        ++imodel;
    }

    if (print_memory == true){
        std::cout << " matrix shape (X): ("
            << total_n_data << "," << n_features << ")" << std::endl;
    }

    x_all = Eigen::MatrixXd::Zero(total_n_data, n_features);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided,1)
    #endif
    for (int i = 0; i < n_st; ++i){
        std::set<int> uniq_types(types[i].begin(), types[i].end());
        for (size_t n = 0; n < cumulative_n_features.size(); ++n){
            int first_index;
            if (n == 0) first_index = 0;
            else first_index = cumulative_n_features[n-1];

            vector1i active_atoms, types_active;
            vector2d positions_c_active;
            find_active_atoms(
                type_full_array[n],
                type_indices_array[n],
                types[i],
                positions_c[i],
                active_atoms,
                types_active,
                positions_c_active
            );

            Neighbor neigh(
                axis[i],
                positions_c_active,
                types_active,
                fp_array[n].n_type,
                fp_array[n].cutoff
            );

            vector1d xe;
            vector2d xf, xs;
            model_array[n].run(
                neigh.get_dis_array(),
                neigh.get_diff_array(),
                neigh.get_atom2_array(),
                types_active,
                force_st[i],
                xe, xf, xs
            );

            for (size_t j = 0; j < xe.size(); ++j)
                x_all(i,first_index+j) = xe[j];

            if (force_st[i] == true){
                for (size_t j = 0; j < xf.size(); ++j) {
                    const auto j_rev = 3 * active_atoms[j / 3] + j % 3;
                    for (size_t k = 0; k < xf[j].size(); ++k){
                        x_all(xf_begin[i] + j_rev, first_index + k) = xf[j][k];
                    }
                }
                for (size_t j = 0; j < xs.size(); ++j) {
                    for (size_t k = 0; k < xs[j].size(); ++k){
                        x_all(xs_begin[i] + j, first_index + k) = xs[j][k];
                    }
                }
            }
        }
    }
}

PyHybridModel::~PyHybridModel(){}

void PyHybridModel::find_active_atoms(
    const bool type_full,
    const vector1i& type_indices,
    const vector1i& types_old,
    const vector2d& positions_c_old,
    vector1i& active_atoms,
    vector1i& types_active,
    vector2d& positions_c_active
){
    active_atoms = vector1i({});
    if (type_full == false){
        types_active = vector1i({});
        int atom(0);
        for (const auto t: types_old){
            auto iter = std::find(type_indices.begin(), type_indices.end(), t);
            if (iter != type_indices.end()) {
                active_atoms.emplace_back(atom);
                types_active.emplace_back(types_old[atom]);
            }
            ++atom;
        }
        positions_c_active = vector2d(3, vector1d(active_atoms.size()));
        int j(0);
        for (const auto atom: active_atoms){
            for (int i = 0; i < 3; ++i){
                positions_c_active[i][j] = positions_c_old[i][active_atoms[j]];
            }
            ++j;
        }
        int t_rep(0);
        for (const auto t: type_indices){
            std::replace(types_active.begin(), types_active.end(), t, t_rep);
            ++t_rep;
        }
    }
    else {
        for (int i = 0; i < positions_c_old[0].size(); ++i)
            active_atoms.emplace_back(i);
        types_active = types_old;
        positions_c_active = positions_c_old;
    }
}

void PyHybridModel::set_index(const std::vector<int>& n_data_dataset,
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

Eigen::MatrixXd& PyHybridModel::get_x(){ return x_all; }

const vector1i& PyHybridModel::get_fbegin() const{ return xf_begin_dataset; }

const vector1i& PyHybridModel::get_sbegin() const{ return xs_begin_dataset; }

const vector1i& PyHybridModel::get_cumulative_n_features() const{
    return cumulative_n_features;
}

const vector1i& PyHybridModel::get_n_data() const{ return n_data; }
