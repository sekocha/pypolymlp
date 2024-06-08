/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/model.h"

Model::Model(){}

Model::Model(const vector3d& dis_array,
             const vector4d& diff_array,
             const vector3i& atom2_array,
             const vector1i& types_i,
             const struct feature_params& fp,
             const bool& element_swap){

    types = types_i;

    n_atom = dis_array.size();
    n_type = fp.n_type;
    force = fp.force;
    model_type = fp.model_type;
    maxp = fp.maxp;
    modelp = ModelParams(fp, element_swap);

    const int size = modelp.get_n_coeff_all();
    xe_sum = vector1d(size, 0.0);
    if (force == true){
        xf_sum = vector2d(3*n_atom, vector1d(size,0.0));
        xs_sum = vector2d(6, vector1d(size,0.0));
    }

    if (fp.des_type == "pair")
        pair(dis_array, diff_array, atom2_array, fp);
    else if (fp.des_type == "gtinv")
        gtinv(dis_array, diff_array, atom2_array, fp);
}

Model::~Model(){}

void Model::pair(const vector3d& dis_array,
                 const vector4d& diff_array,
                 const vector3i& atom2_array,
                 const struct feature_params& fp){

    //#ifdef _OPENMP
    //#pragma omp parallel for //reduction(+:xe_sum, xf_sum, xs_sum)
    //#endif
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        vector1d de; vector2d dfx, dfy, dfz, ds;
        Local local(n_atom, atom1, types[atom1], fp, modelp);
        if (force == false) de = local.pair(dis_array[atom1]);
        else local.pair_d(dis_array[atom1],
                          diff_array[atom1],
                          atom2_array[atom1],
                          de, dfx, dfy, dfz, ds);
        model_common(de, dfx, dfy, dfz, ds, types[atom1]);
    }
}

void Model::gtinv(const vector3d& dis_array,
                  const vector4d& diff_array,
                  const vector3i& atom2_array,
                  const struct feature_params& fp){

    //#ifdef _OPENMP
    //#pragma omp parallel for //reduction(+:xe_sum, xf_sum, xs_sum)
    //#endif
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        vector1d de; vector2d dfx, dfy, dfz, ds;
        Local local(n_atom, atom1, types[atom1], fp, modelp);
        if (force == false) {
            de = local.gtinv(dis_array[atom1], diff_array[atom1]);
        }
        else {
            local.gtinv_d(dis_array[atom1],
                          diff_array[atom1],
                          atom2_array[atom1],
                          de, dfx, dfy, dfz, ds);
        }
        model_common(de, dfx, dfy, dfz, ds, types[atom1]);
    }
}

void Model::model_common(const vector1d& de,
                         const vector2d& dfx,
                         const vector2d& dfy,
                         const vector2d& dfz,
                         const vector2d& ds,
                         const int& type1){

    int col = 0;
    model_linear(de, dfx, dfy, dfz, ds, col);
    if (model_type == 1 and maxp > 1) model1(de, dfx, dfy, dfz, ds, col);
    else if (model_type > 1){
        if (maxp > 1) model2_comb2(de, dfx, dfy, dfz, ds, col);
        if (maxp > 2) model2_comb3(de, dfx, dfy, dfz, ds, col);
    }
}

void Model::model_linear(const vector1d& de,
                         const vector2d& dfx,
                         const vector2d& dfy,
                         const vector2d& dfz,
                         const vector2d& ds,
                         int& col){

    const int n_linear = de.size();
    for (int n = 0; n < n_linear; ++n){
        xe_sum[col] += de[n];
        if (force == true){
            for (int k = 0; k < n_atom; ++k){
                xf_sum[3*k][col] += dfx[n][k];
                xf_sum[3*k+1][col] += dfy[n][k];
                xf_sum[3*k+2][col] += dfz[n][k];
            }
            for (int k = 0; k < 6; ++k) xs_sum[k][col] += ds[n][k];
        }
        ++col;
    }
}

void Model::model1(const vector1d& de,
                   const vector2d& dfx,
                   const vector2d& dfy,
                   const vector2d& dfz,
                   const vector2d& ds,
                   int& col){

    const int n_linear = de.size();
    double val;
    for (int p = 2; p < maxp + 1; ++p){
        for (int n = 0; n < n_linear; ++n){
            xe_sum[col] += pow(de[n], p);
            if (force == true){
                val = p * pow(de[n], p-1);
                for (size_t k = 0; k < dfx[n].size(); ++k){
                    xf_sum[3*k][col] += val * dfx[n][k];
                    xf_sum[3*k+1][col] += val * dfy[n][k];
                    xf_sum[3*k+2][col] += val * dfz[n][k];
                }
                for (int k = 0; k < 6; ++k) xs_sum[k][col] += val * ds[n][k];
            }
            ++col;
        }
    }
}

void Model::model2_comb2(const vector1d& de,
                         const vector2d& dfx,
                         const vector2d& dfy,
                         const vector2d& dfz,
                         const vector2d& ds,
                         int& col){

    int c1, c2;
    double val1, val2;
    for (const auto& comb: modelp.get_comb2()){
        c1 = comb[0], c2 = comb[1];
        xe_sum[col] += de[c1] * de[c2];
        if (force == true){
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
        ++col;
    }
}

void Model::model2_comb3(const vector1d& de,
                         const vector2d& dfx,
                         const vector2d& dfy,
                         const vector2d& dfz,
                         const vector2d& ds,
                         int& col){

    int c1, c2, c3;
    double val1, val2, val3;
    for (const auto& comb: modelp.get_comb3()){
        c1 = comb[0], c2 = comb[1], c3 = comb[2];
        xe_sum[col] += de[c1] * de[c2] * de[c3];
        if (force == true){
            val1 = de[c2] * de[c3];
            val2 = de[c1] * de[c3];
            val3 = de[c1] * de[c2];
            for (size_t k = 0; k < dfx[c1].size(); ++k){
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
        ++col;
    }
}

const vector1d& Model::get_xe_sum() const{ return xe_sum;}
const vector2d& Model::get_xf_sum() const{ return xf_sum;}
const vector2d& Model::get_xs_sum() const{ return xs_sum;}
