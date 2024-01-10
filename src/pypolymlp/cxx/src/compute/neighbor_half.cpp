/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_half.h"

NeighborHalf::NeighborHalf(const vector2d& axis, 
                           const vector2d& positions_c, 
                           const vector1i& types, 
                           const double& cutoff){

    const double tol = 1e-12;
    const auto& trans = find_trans(axis, cutoff);

    const int n_total_atom = types.size();
    half_list = vector2i(n_total_atom);
    diff_list = vector3d(n_total_atom);

    double dx, dy, dz;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < n_total_atom; ++j){
            for (const auto& tr: trans){
                dx = positions_c[0][j] + tr[0] - positions_c[0][i];
                dy = positions_c[1][j] + tr[1] - positions_c[1][i];
                dz = positions_c[2][j] + tr[2] - positions_c[2][i];
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                bool bool_half = false;
                if (dis < cutoff and dis > 1e-10){
                    if (dz >= tol) 
                        bool_half = true;
                    else if (fabs(dz) < tol and dy >= tol) 
                        bool_half = true;
                    else if (fabs(dz) < tol and fabs(dy) < tol and dx >=tol) 
                        bool_half = true;
                }

                if (bool_half == true){
                    half_list[i].emplace_back(j);
                    diff_list[i].emplace_back(vector1d({dx, dy, dz}));
                }
            }
        }
    }
}

NeighborHalf::~NeighborHalf(){}

vector1d NeighborHalf::prod(const vector2d& mat, const vector1i& vec){

    vector1d res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i){
        for (size_t j = 0; j < mat[i].size(); ++j){
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}

vector2d NeighborHalf::find_trans(const vector2d& axis, const double& cutoff){

    int m = 10;
    vector1i max_exp(3,1);
    for (int i = 0; i < m + 1; ++i){
        for (int j = 0; j < m + 1; ++j){
            for (int k = 0; k < m + 1; ++k){
                vector1i vec = {i,j,k};
                vector1d vec_c = prod(axis, vec);
                double dis = sqrt(vec_c[0]*vec_c[0]
                    +vec_c[1]*vec_c[1]+vec_c[2]*vec_c[2]);
                if (dis > 0 and dis < cutoff){
                    double exp = ceil(cutoff/dis);
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

const vector2i& NeighborHalf::get_half_list() const{ return half_list; }
const vector3d& NeighborHalf::get_diff_list() const{ return diff_list; }

