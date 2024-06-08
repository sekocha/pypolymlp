/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/model_properties.h"

ModelProperties::ModelProperties(){}

ModelProperties::ModelProperties(const vector3d& dis_array,
                                 const vector4d& diff_array,
                                 const vector3i& atom2_array,
                                 const vector1i& types_i,
                                 const vector1d& coeffs_i,
                                 const struct feature_params& fp,
                                 const ModelParams& modelp,
                                 const FunctionFeatures& features){

    types = types_i;

    n_atom = dis_array.size();
    n_type = fp.n_type;
    use_force = fp.force;
    model_type = fp.model_type;
    maxp = fp.maxp;
    coeffs = coeffs_i;

    energy = 0.0;
    if (use_force == true){
        force = vector1d(3*n_atom, 0.0);
        stress = vector1d(6, 0.0);
    }

    if (fp.des_type == "pair")
        pair(dis_array, diff_array, atom2_array, fp);
    else if (fp.des_type == "gtinv")
        gtinv(dis_array, diff_array, atom2_array, fp);
}

ModelProperties::~ModelProperties(){}

void ModelProperties::pair(const vector3d& dis_array,
                           const vector4d& diff_array,
                           const vector3i& atom2_array,
                           const struct feature_params& fp,
                           const ModelParams& modelp,
                           const FunctionFeatures& features){

    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        vector1d de; vector2d dfx, dfy, dfz, ds;
        LocalFast local(n_atom, atom1, types[atom1], fp, modelp);
        if (use_force == false) local.pair(dis_array[atom1], de);
        else local.pair_d(dis_array[atom1],
                          diff_array[atom1],
                          atom2_array[atom1],
                          de, dfx, dfy, dfz, ds);
        model_common(de, dfx, dfy, dfz, ds, types[atom1]);
    }
}

void ModelProperties::gtinv(const vector3d& dis_array,
                            const vector4d& diff_array,
                            const vector3i& atom2_array,
                            const struct feature_params& fp,
                            const ModelParams& modelp,
                            const FunctionFeatures& features){

    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        vector1d de; vector2d dfx, dfy, dfz, ds;
        LocalFast local(n_atom, atom1, types[atom1], fp, modelp);
        if (use_force == false) {
            local.gtinv(dis_array[atom1], diff_array[atom1], de);
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

void ModelProperties::model_common(const vector1d& de,
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

void ModelProperties::model_linear(const vector1d& de,
                                  const vector2d& dfx,
                                  const vector2d& dfy,
                                  const vector2d& dfz,
                                  const vector2d& ds,
                                  int& col){

    const int n_linear = de.size();
    for (int n = 0; n < n_linear; ++n){
        energy += de[n] * coeffs[col];
        if (use_force == true){
            for (int k = 0; k < n_atom; ++k){
                force[3*k] += dfx[n][k] * coeffs[col];
                force[3*k+1] += dfy[n][k] * coeffs[col];
                force[3*k+2] += dfz[n][k] * coeffs[col];
            }
            for (int k = 0; k < 6; ++k)
                stress[k] += ds[n][k] * coeffs[col];
        }
        ++col;
    }
}

void ModelProperties::model1(const vector1d& de,
                             const vector2d& dfx,
                             const vector2d& dfy,
                             const vector2d& dfz,
                             const vector2d& ds,
                             int& col){

    const int n_linear = de.size();
    double val;
    for (int p = 2; p < maxp + 1; ++p){
        for (int n = 0; n < n_linear; ++n){
            energy += pow(de[n], p) * coeffs[col];
            if (use_force == true){
                val = p * pow(de[n], p-1) * coeffs[col];
                for (size_t k = 0; k < dfx[n].size(); ++k){
                    force[3*k] += val * dfx[n][k];
                    force[3*k+1] += val * dfy[n][k];
                    force[3*k+2] += val * dfz[n][k];
                }
                for (int k = 0; k < 6; ++k)
                    stress[k] += val * ds[n][k];
            }
            ++col;
        }
    }
}

void ModelProperties::model2_comb2(const vector1d& de,
                                   const vector2d& dfx,
                                   const vector2d& dfy,
                                   const vector2d& dfz,
                                   const vector2d& ds,
                                   int& col){

    int c1, c2;
    double val1, val2;
    for (const auto& comb: modelp.get_comb2()){
        c1 = comb[0], c2 = comb[1];
        energy += de[c1] * de[c2] * coeffs[col];
        if (use_force == true){
            val1 = de[c2] * coeffs[col], val2 = de[c1] * coeffs[col];
            for (int k = 0; k < n_atom; ++k){
                force[3*k] += val1 * dfx[c1][k] + val2 * dfx[c2][k];
                force[3*k+1] += val1 * dfy[c1][k] + val2 * dfy[c2][k];
                force[3*k+2] += val1 * dfz[c1][k] + val2 * dfz[c2][k];
            }
            for (int k = 0; k < 6; ++k){
                stress[k] += val1 * ds[c1][k] + val2 * ds[c2][k];
            }
        }
        ++col;
    }
}

void ModelProperties::model2_comb3(const vector1d& de,
                                   const vector2d& dfx,
                                   const vector2d& dfy,
                                   const vector2d& dfz,
                                   const vector2d& ds,
                                   int& col){

    int c1, c2, c3;
    double val1, val2, val3;
    for (const auto& comb: modelp.get_comb3()){
        c1 = comb[0], c2 = comb[1], c3 = comb[2];
        energy += de[c1] * de[c2] * de[c3] * coeffs[col];
        if (use_force == true){
            val1 = de[c2] * de[c3] * coeffs[col];
            val2 = de[c1] * de[c3] * coeffs[col];
            val3 = de[c1] * de[c2] * coeffs[col];
            for (size_t k = 0; k < dfx[c1].size(); ++k){
                force[3*k] += val1 * dfx[c1][k]
                            + val2 * dfx[c2][k] + val3 * dfx[c3][k];
                force[3*k+1] += val1 * dfy[c1][k]
                            + val2 * dfy[c2][k] + val3 * dfy[c3][k];
                force[3*k+2] += val1 * dfz[c1][k]
                            + val2 * dfz[c2][k] + val3 * dfz[c3][k];
            }
            for (int k = 0; k < 6; ++k){
                stress[k] += val1 * ds[c1][k]
                            + val2 * ds[c2][k] + val3 * ds[c3][k];
            }
        }
        ++col;
    }
}

const double& ModelProperties::get_energy() const{ return energy;}
const vector1d& ModelProperties::get_force() const{ return force;}
const vector1d& ModelProperties::get_stress() const{ return stress;}
