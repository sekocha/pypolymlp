/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/model_fast.h"

ModelFast::ModelFast(){}

ModelFast::ModelFast(const vector3d& dis_array,
                     const vector4d& diff_array,
                     const vector3i& atom2_array,
                     const vector1i& types_i,
                     const struct feature_params& fp,
                     const FunctionFeatures& features){

    types = types_i;
    n_atom = dis_array.size();
    n_type = fp.n_type;
    force = fp.force;
    model_type = fp.model_type;
    maxp = fp.maxp;

    const auto& modelp = features.get_model_params();
    n_linear_features = modelp.get_n_linear_features();
    const int size = modelp.get_n_coeff_all();
    xe_sum = vector1d(size, 0.0);
    if (force == true){
        xf_sum = vector2d(3*n_atom, vector1d(size,0.0));
        xs_sum = vector2d(6, vector1d(size,0.0));
    }

    if (fp.feature_type == "pair")
        pair(dis_array, diff_array, atom2_array, fp, features);
    else if (fp.feature_type == "gtinv")
        gtinv(dis_array, diff_array, atom2_array, fp, features);
}

ModelFast::~ModelFast(){}

void ModelFast::pair(const vector3d& dis_array,
                     const vector4d& diff_array,
                     const vector3i& atom2_array,
                     const struct feature_params& fp,
                     const FunctionFeatures& features){

    const auto& mapping = features.get_mapping();
    //#ifdef _OPENMP
    //#pragma omp parallel for //reduction(+:xe_sum, xf_sum, xs_sum)
    //#endif
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        vector1d de; vector2d dfx, dfy, dfz, ds;
        LocalFast local(n_atom, atom1, types[atom1], fp, features);
        if (force == false) local.pair(dis_array[atom1], features, de);
        else {
            local.pair_d(
                dis_array[atom1], diff_array[atom1], atom2_array[atom1], features,
                de, dfx, dfy, dfz, ds
            );
        }
        model_common(de, dfx, dfy, dfz, ds, features, types[atom1]);
    }
}

void ModelFast::gtinv(const vector3d& dis_array,
                      const vector4d& diff_array,
                      const vector3i& atom2_array,
                      const struct feature_params& fp,
                      const FunctionFeatures& features){

    const auto& mapping = features.get_mapping();
    //#ifdef _OPENMP
    //#pragma omp parallel for //reduction(+:xe_sum, xf_sum, xs_sum)
    //#endif
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        vector1d de; vector2d dfx, dfy, dfz, ds;
        LocalFast local(n_atom, atom1, types[atom1], fp, features);
        if (force == false) {
            local.gtinv(
                dis_array[atom1], diff_array[atom1], features, de
            );
        }
        else {
            local.gtinv_d(
                dis_array[atom1], diff_array[atom1], atom2_array[atom1],
                features, de, dfx, dfy, dfz, ds
            );
        }
        model_common(de, dfx, dfy, dfz, ds, features, types[atom1]);
    }
}

void ModelFast::model_common(const vector1d& de,
                             const vector2d& dfx,
                             const vector2d& dfy,
                             const vector2d& dfz,
                             const vector2d& ds,
                             const FunctionFeatures& features,
                             const int type1){

    model_linear(de, dfx, dfy, dfz, ds, features, type1);
    if (model_type == 1 and maxp > 1) {
        model1(de, dfx, dfy, dfz, ds, features, type1);
    }
    else if (model_type > 1){
        if (maxp > 1) model2_comb2(de, dfx, dfy, dfz, ds, features, type1);
        if (maxp > 2) model2_comb3(de, dfx, dfy, dfz, ds, features, type1);
    }
}

void ModelFast::model_linear(const vector1d& de,
                             const vector2d& dfx,
                             const vector2d& dfy,
                             const vector2d& dfz,
                             const vector2d& ds,
                             const FunctionFeatures& features,
                             const int type1){

    const auto& poly = features.get_polynomial1(type1);
    for (size_t tlocal = 0; tlocal < poly.size(); ++tlocal){
        const auto& pterm = poly[tlocal];
        xe_sum[pterm.seq_id] += de[tlocal];
    }
    if (force == true){
        for (size_t tlocal = 0; tlocal < poly.size(); ++tlocal){
            const auto& pterm = poly[tlocal];
            for (int k = 0; k < n_atom; ++k){
                xf_sum[3*k][pterm.seq_id] += dfx[tlocal][k];
                xf_sum[3*k+1][pterm.seq_id] += dfy[tlocal][k];
                xf_sum[3*k+2][pterm.seq_id] += dfz[tlocal][k];
            }
            for (int k = 0; k < 6; ++k)
                xs_sum[k][pterm.seq_id] += ds[tlocal][k];
        }
    }
}

void ModelFast::model1(const vector1d& de,
                       const vector2d& dfx,
                       const vector2d& dfy,
                       const vector2d& dfz,
                       const vector2d& ds,
                       const FunctionFeatures& features,
                       const int type1){

    const auto& poly = features.get_polynomial1(type1);
    int col;
    double val;
    for (int p = 2; p < maxp + 1; ++p){
        for (size_t tlocal = 0; tlocal < poly.size(); ++tlocal){
            const auto& pterm = poly[tlocal];
            col = n_linear_features * (p - 1) + pterm.seq_id;
            xe_sum[col] += pow(de[tlocal], p);
            if (force == true){
                val = p * pow(de[tlocal], p-1);
                for (int k = 0; k < n_atom; ++k){
                    xf_sum[3*k][col] += val * dfx[tlocal][k];
                    xf_sum[3*k+1][col] += val * dfy[tlocal][k];
                    xf_sum[3*k+2][col] += val * dfz[tlocal][k];
                }
                for (int k = 0; k < 6; ++k)
                    xs_sum[k][col] += val * ds[tlocal][k];
            }
        }
    }
}

void ModelFast::model2_comb2(const vector1d& de,
                             const vector2d& dfx,
                             const vector2d& dfy,
                             const vector2d& dfz,
                             const vector2d& ds,
                             const FunctionFeatures& features,
                             const int type1){

    int col, c1, c2;
    double val1, val2;
    const auto& poly = features.get_polynomial2(type1);
    for (const auto& pterm: poly){
        col = pterm.seq_id;
        c1 = pterm.comb_tlocal[0], c2 = pterm.comb_tlocal[1];
        xe_sum[col] += de[c1] * de[c2];
    }
    if (force == true){
        for (const auto& pterm: poly){
            col = pterm.seq_id;
            c1 = pterm.comb_tlocal[0], c2 = pterm.comb_tlocal[1];
            val1 = de[c2], val2 = de[c1];
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
}

void ModelFast::model2_comb3(const vector1d& de,
                             const vector2d& dfx,
                             const vector2d& dfy,
                             const vector2d& dfz,
                             const vector2d& ds,
                             const FunctionFeatures& features,
                             const int type1){

    int col, c1, c2, c3;
    double val1, val2, val3;
    const auto& poly = features.get_polynomial3(type1);
    for (const auto& pterm: poly){
        col = pterm.seq_id;
        c1 = pterm.comb_tlocal[0];
        c2 = pterm.comb_tlocal[1];
        c3 = pterm.comb_tlocal[2];
        xe_sum[col] += de[c1] * de[c2] * de[c3];
    }
    if (force == true){
        for (const auto& pterm: poly){
            col = pterm.seq_id;
            c1 = pterm.comb_tlocal[0];
            c2 = pterm.comb_tlocal[1];
            c3 = pterm.comb_tlocal[2];
            val1 = de[c2] * de[c3];
            val2 = de[c1] * de[c3];
            val3 = de[c1] * de[c2];
            for (int k = 0; k < n_atom; ++k){
                xf_sum[3*k][col] += val1 * dfx[c1][k]
                        + val2 * dfx[c2][k] + val3 * dfx[c3][k];
                xf_sum[3*k+1][col] += val1 * dfy[c1][k]
                        + val2 * dfy[c2][k] + val3 * dfy[c3][k];
                xf_sum[3*k+2][col] += val1 * dfz[c1][k]
                        + val2 * dfz[c2][k] + val3 * dfz[c3][k];
            }
            for (int k = 0; k < 6; ++k){
                xs_sum[k][col] += val1 * ds[c1][k]
                        + val2 * ds[c2][k] + val3 * ds[c3][k];
            }
        }
    }
}

const vector1d& ModelFast::get_xe_sum() const{ return xe_sum;}
const vector2d& ModelFast::get_xf_sum() const{ return xf_sum;}
const vector2d& ModelFast::get_xs_sum() const{ return xs_sum;}
