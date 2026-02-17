/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/model.h"
#include <chrono>


Model::Model(){}

Model::Model(const struct feature_params& fp){

    polymlp.set_features(fp);
}

Model::~Model(){}


void Model::run(
    const vector3d& dis_array,
    const vector4d& diff_array,
    const vector3i& atom2_array,
    const vector1i& types,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    const auto& fp = polymlp.get_fp();
    const int size = polymlp.get_n_variables();
    xe_sum = Eigen::VectorXd::Zero(size);
    if (force == true){
        const int n_atom = types.size();
        xf_sum = Eigen::MatrixXd::Zero(3 * n_atom, size);
        xs_sum = Eigen::MatrixXd::Zero(6, size);
    }
    if (fp.feature_type == "pair")
        pair(dis_array, diff_array, atom2_array, types, force, xe_sum, xf_sum, xs_sum);
    else if (fp.feature_type == "gtinv")
        gtinv(dis_array, diff_array, atom2_array, types, force, xe_sum, xf_sum, xs_sum);
}


void Model::pair(
    const vector3d& dis_array,
    const vector4d& diff_array,
    const vector3i& atom2_array,
    const vector1i& types,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    const int n_atom = types.size();
    LocalPair local(n_atom);
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        const int type1 = types[atom1];
        const auto& dis = dis_array[atom1];
        vector1d de; vector2d dfx, dfy, dfz, ds;
        if (force == false) {
            local.pair(polymlp, type1, dis, de);
        }
        else {
            const auto& diff = diff_array[atom1];
            const auto& atom2 = atom2_array[atom1];
            local.pair_d(
                polymlp, atom1, type1, dis, diff, atom2, de, dfx, dfy, dfz, ds);
        }

        Eigen::VectorXd de_eig;
        Eigen::MatrixXd df_eig, ds_eig;
        reshape(de, dfx, dfy, dfz, ds, force, de_eig, df_eig, ds_eig);

        model_polynomial(
            de_eig, df_eig, ds_eig,
            type1, force, xe_sum, xf_sum, xs_sum);
    }
}


void Model::gtinv(
    const vector3d& dis_array,
    const vector4d& diff_array,
    const vector3i& atom2_array,
    const vector1i& types,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    const int n_atom = types.size();
    Local local(n_atom);
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        const int type1 = types[atom1];
        const auto& dis = dis_array[atom1];
        const auto& diff = diff_array[atom1];
        vector1d de; vector2d dfx, dfy, dfz, ds;
        auto t1 = std::chrono::high_resolution_clock::now();
        if (force == false) {
            local.gtinv(polymlp, type1, dis, diff, de);
        }
        else {
            const auto& atom2 = atom2_array[atom1];
            local.gtinv_d(
                polymlp, atom1, type1, dis, diff, atom2, de, dfx, dfy, dfz, ds);
        }

        Eigen::VectorXd de_eig;
        Eigen::MatrixXd df_eig, ds_eig;
        reshape(de, dfx, dfy, dfz, ds, force, de_eig, df_eig, ds_eig);

        auto t2 = std::chrono::high_resolution_clock::now();
        model_polynomial(
            de_eig, df_eig, ds_eig,
            type1, force, xe_sum, xf_sum, xs_sum);
        auto t3 = std::chrono::high_resolution_clock::now();

        //auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
        //std::cout << "tm1: " << duration1.count() << " micros" << std::endl;
        //std::cout << "tm2: " << duration2.count() << " micros" << std::endl;
   }
}


void Model::reshape(
    const vector1d& de,
    const vector2d& dfx,
    const vector2d& dfy,
    const vector2d& dfz,
    const vector2d& ds,
    const bool force,
    Eigen::VectorXd& de_eig,
    Eigen::MatrixXd& df_eig,
    Eigen::MatrixXd& ds_eig){

    de_eig = Eigen::VectorXd(de.size());
    if (force){
        const int three_n = 3 * dfx[0].size();
        df_eig = Eigen::MatrixXd(three_n, dfx.size());
        ds_eig = Eigen::MatrixXd(6, ds.size());
    }

    for (size_t i = 0; i < de.size(); ++i)
        de_eig(i) = de[i];

    if (!force) return;

    for (size_t i = 0; i < dfx.size(); ++i)
        for (size_t j = 0; j < dfx[i].size(); ++j)
            df_eig(3 * j, i) = dfx[i][j];

    for (size_t i = 0; i < dfy.size(); ++i)
        for (size_t j = 0; j < dfy[i].size(); ++j)
            df_eig(3 * j + 1, i) = dfy[i][j];

    for (size_t i = 0; i < dfz.size(); ++i)
        for (size_t j = 0; j < dfz[i].size(); ++j)
            df_eig(3 * j + 2, i) = dfz[i][j];

    for (size_t i = 0; i < ds.size(); ++i)
        for (size_t j = 0; j < ds[i].size(); ++j)
            ds_eig(j, i) = ds[i][j];

}



void Model::model_polynomial(
    const Eigen::VectorXd& de,
    const Eigen::MatrixXd& df,
    const Eigen::MatrixXd& ds,
    const int type1,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    auto& maps = polymlp.get_maps();
    auto& maps_type = maps.maps_type[type1];
    for (const auto& term: maps_type.polynomial){
        if (term.local_ids.size() == 1)
            model_order1(term, de, df, ds, force, xe_sum, xf_sum, xs_sum);
        else if (term.local_ids.size() == 2)
            model_order2(term, de, df, ds, force, xe_sum, xf_sum, xs_sum);
        else if (term.local_ids.size() == 3)
            model_order3(term, de, df, ds, force, xe_sum, xf_sum, xs_sum);
    }
}


void Model::model_order1(
    const PolynomialTerm& term,
    const Eigen::VectorXd& de,
    const Eigen::MatrixXd& df,
    const Eigen::MatrixXd& ds,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    const int col = term.global_id;
    const int c1 = term.local_ids[0];

    xe_sum(col) += de(c1);

    if (!force) return;

    const int n_atom_3 = df.rows();
    for (int k = 0; k < n_atom_3; ++k){
        xf_sum(k, col) += df(k, c1);
    }
    for (int k = 0; k < 6; ++k){
        xs_sum(k, col) += ds(k, c1);
    }
}

void Model::model_order2(
    const PolynomialTerm& term,
    const Eigen::VectorXd& de,
    const Eigen::MatrixXd& df,
    const Eigen::MatrixXd& ds,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    const int col = term.global_id;
    const int c1 = term.local_ids[0];
    const int c2 = term.local_ids[1];

    xe_sum(col) += de(c1) * de(c2);

    if (!force) return;

    const double val1 = de(c2);
    const double val2 = de(c1);

    const int n_atom_3 = df.rows();
    for (int k = 0; k < n_atom_3; ++k){
        xf_sum(k, col) += val1 * df(k, c1) + val2 * df(k, c2);
    }
    for (int k = 0; k < 6; ++k){
        xs_sum(k, col) += val1 * ds(k, c1) + val2 * ds(k, c2);
    }
}


void Model::model_order3(
    const PolynomialTerm& term,
    const Eigen::VectorXd& de,
    const Eigen::MatrixXd& df,
    const Eigen::MatrixXd& ds,
    const bool force,
    Eigen::VectorXd& xe_sum,
    Eigen::MatrixXd& xf_sum,
    Eigen::MatrixXd& xs_sum){

    const int col = term.global_id;
    const int c1 = term.local_ids[0];
    const int c2 = term.local_ids[1];
    const int c3 = term.local_ids[2];

    xe_sum(col) += de(c1) * de(c2) * de(c3);

    if (!force) return;

    const double val1 = de(c2) * de(c3);
    const double val2 = de(c1) * de(c3);
    const double val3 = de(c1) * de(c2);

    const int n_atom_3 = df.rows();
    for (int k = 0; k < n_atom_3; ++k){
        xf_sum(k, col) += val1 * df(k, c1) + val2 * df(k, c2) + val3 * df(k, c3);
    }
    for (int k = 0; k < 6; ++k){
        xs_sum(k, col) += val1 * ds(k, c1) + val2 * ds(k, c2) + val3 * ds(k, c3);
    }
}

int Model::get_n_features(){
    return polymlp.get_n_variables();
}
