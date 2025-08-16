/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/model.h"


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
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){

    const auto& fp = polymlp.get_fp();
    const int size = polymlp.get_n_variables();
    xe_sum = vector1d(size, 0.0);
    if (force == true){
        const int n_atom = types.size();
        xf_sum = vector2d(3 * n_atom, vector1d(size, 0.0));
        xs_sum = vector2d(6, vector1d(size, 0.0));
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
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){

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
            local.pair_d(polymlp, atom1, type1, dis, diff, atom2, de, dfx, dfy, dfz, ds);
        }
        model_polynomial(de, dfx, dfy, dfz, ds, type1, force, xe_sum, xf_sum, xs_sum);
    }
}


void Model::gtinv(
    const vector3d& dis_array,
    const vector4d& diff_array,
    const vector3i& atom2_array,
    const vector1i& types,
    const bool force,
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){

    const int n_atom = types.size();
    Local local(n_atom);
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        const int type1 = types[atom1];
        const auto& dis = dis_array[atom1];
        const auto& diff = diff_array[atom1];
        vector1d de; vector2d dfx, dfy, dfz, ds;
        if (force == false) {
            local.gtinv(polymlp, type1, dis, diff, de);
        }
        else {
            const auto& atom2 = atom2_array[atom1];
            local.gtinv_d(polymlp, atom1, type1, dis, diff, atom2, de, dfx, dfy, dfz, ds);
        }
        model_polynomial(de, dfx, dfy, dfz, ds, type1, force, xe_sum, xf_sum, xs_sum);
   }
}


void Model::model_polynomial(
    const vector1d& de,
    const vector2d& dfx,
    const vector2d& dfy,
    const vector2d& dfz,
    const vector2d& ds,
    const int type1,
    const bool force,
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){

    auto& maps = polymlp.get_maps();
    auto& maps_type = maps.maps_type[type1];
    for (const auto& term: maps_type.polynomial){
        if (term.local_ids.size() == 1)
            model_order1(term, de, dfx, dfy, dfz, ds, force, xe_sum, xf_sum, xs_sum);
        else if (term.local_ids.size() == 2)
            model_order2(term, de, dfx, dfy, dfz, ds, force, xe_sum, xf_sum, xs_sum);
        else if (term.local_ids.size() == 3)
            model_order3(term, de, dfx, dfy, dfz, ds, force, xe_sum, xf_sum, xs_sum);
    }
}


void Model::model_order1(
    const PolynomialTerm& term,
    const vector1d& de,
    const vector2d& dfx,
    const vector2d& dfy,
    const vector2d& dfz,
    const vector2d& ds,
    const bool force,
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){
    const int col = term.global_id;
    const int c1 = term.local_ids[0];

    xe_sum[col] += de[c1];
    if (force == true){
        const int n_atom = dfx[0].size();
        for (int k = 0; k < n_atom; ++k){
            xf_sum[3*k][col] += dfx[c1][k];
            xf_sum[3*k+1][col] += dfy[c1][k];
            xf_sum[3*k+2][col] += dfz[c1][k];
        }
        for (int k = 0; k < 6; ++k){
            xs_sum[k][col] += ds[c1][k];
        }
    }
}

void Model::model_order2(
    const PolynomialTerm& term,
    const vector1d& de,
    const vector2d& dfx,
    const vector2d& dfy,
    const vector2d& dfz,
    const vector2d& ds,
    const bool force,
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){
    const int col = term.global_id;
    const int c1 = term.local_ids[0];
    const int c2 = term.local_ids[1];

    xe_sum[col] += de[c1] * de[c2];
    if (force == true){
        const int n_atom = dfx[0].size();
        const double val1 = de[c2];
        const double val2 = de[c1];
        for (int k = 0; k < n_atom; ++k){
            xf_sum[3*k][col] += val1 * dfx[c1][k] + val2 * dfx[c2][k];
            xf_sum[3*k+1][col] += val1 * dfy[c1][k] + val2 * dfy[c2][k];
            xf_sum[3*k+2][col] += val1 * dfz[c1][k] + val2 * dfz[c2][k];
        }
        for (int k = 0; k < 6; ++k){
            xs_sum[k][col] += val1 * ds[c1][k] + val2 * ds[c2][k];
        }
    }
}

void Model::model_order3(
    const PolynomialTerm& term,
    const vector1d& de,
    const vector2d& dfx,
    const vector2d& dfy,
    const vector2d& dfz,
    const vector2d& ds,
    const bool force,
    vector1d& xe_sum,
    vector2d& xf_sum,
    vector2d& xs_sum
){
    const int col = term.global_id;
    const int c1 = term.local_ids[0];
    const int c2 = term.local_ids[1];
    const int c3 = term.local_ids[2];

    xe_sum[col] += de[c1] * de[c2] * de[c3];
    if (force == true){
        const int n_atom = dfx[0].size();
        const double val1 = de[c2] * de[c3];
        const double val2 = de[c1] * de[c3];
        const double val3 = de[c1] * de[c2];
        for (int k = 0; k < n_atom; ++k){
            xf_sum[3*k][col] += val1 * dfx[c1][k]
                             + val2 * dfx[c2][k]
                             + val3 * dfx[c3][k];
            xf_sum[3*k+1][col] += val1 * dfy[c1][k]
                               + val2 * dfy[c2][k]
                               + val3 * dfy[c3][k];
            xf_sum[3*k+2][col] += val1 * dfz[c1][k]
                               + val2 * dfz[c2][k]
                               + val3 * dfz[c3][k];
        }
        for (int k = 0; k < 6; ++k){
            xs_sum[k][col] += val1 * ds[c1][k]
                + val2 * ds[c2][k] + val3 * ds[c3][k];
        }
    }
}
