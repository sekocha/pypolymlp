/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor.h"

Neighbor::Neighbor(const vector2d& axis, 
                   const vector2d& positions_c, 
                   const vector1i& types, 
                   const int& n_type, 
                   const double& cutoff){

    const auto& trans = find_trans(axis, cutoff);

    const int n_total_atom = types.size();
    dis_array = vector3d(n_total_atom, vector2d(n_type));
    diff_array = vector4d(n_total_atom, vector3d(n_type));
    atom2_array = vector3i(n_total_atom, vector2i(n_type));

    double dx, dy, dz;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < n_total_atom; ++j){
            for (const auto& tr: trans){
                dx = positions_c[0][j] + tr[0] - positions_c[0][i];
                dy = positions_c[1][j] + tr[1] - positions_c[1][i];
                dz = positions_c[2][j] + tr[2] - positions_c[2][i];
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    int type2 = types[j];
                    dis_array[i][type2].emplace_back(dis);
                    diff_array[i][type2].emplace_back(vector1d{dx,dy,dz});
                    atom2_array[i][type2].emplace_back(j);
                }
            }
        }
    }
}

Neighbor::~Neighbor(){}

vector1d Neighbor::prod(const vector2d& mat, const vector1i& vec){

    vector1d res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i){
        for (size_t j = 0; j < mat[i].size(); ++j){
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}

vector2d Neighbor::find_trans(const vector2d& axis, const double& cutoff){

    int m = 10;
    vector1i max_exp(3,1);
    for (int i = 0; i < m + 1; ++i){
        for (int j = 0; j < m + 1; ++j){
            for (int k = 0; k < m + 1; ++k){
                vector1i vec = {i,j,k};
                vector1d vec_c = prod(axis, vec);
                double dis = sqrt(vec_c[0]*vec_c[0]
                    +vec_c[1]*vec_c[1]+vec_c[2]*vec_c[2]);
                if (dis > 0 and dis < 2 * cutoff){
                    double exp = ceil(2 * cutoff/dis);
                    for (int l = 0; l < 3; ++l){
                        if (exp * vec[l] > max_exp[l]) 
                            max_exp[l] = exp * vec[l];
                    }
                }
            }
        }
    }

    vector2d trans_c_array;
    for (int i = -max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = -max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = -max_exp[2]; k < max_exp[2] + 1; ++k){
                trans_c_array.emplace_back(prod(axis, vector1i{i,j,k}));
            }
        }
    }
    return trans_c_array;
}

const vector3d& Neighbor::get_dis_array() const{ return dis_array; }
const vector4d& Neighbor::get_diff_array() const{ return diff_array; }
const vector3i& Neighbor::get_atom2_array() const{ return atom2_array; }

