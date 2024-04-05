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

double Neighbor::distance(const vector2d& axis, 
                          const int i, const int j, const int k){
    vector1i vec = {i,j,k};
    vector1d vec_c = prod(axis, vec);
    double dis = sqrt(vec_c[0]*vec_c[0]
                    +vec_c[1]*vec_c[1]
                    +vec_c[2]*vec_c[2]);
    return dis;
}

vector2d Neighbor::find_trans(const vector2d& axis, const double& cutoff){

    vector2i vertices;
    for (int i = 0; i < 2; ++i){
        for (int j = 0; j < 2; ++j){
            for (int k = 0; k < 2; ++k){
                vertices.emplace_back(vector1i{i,j,k});
            }
        }
    }

    double max_length(0.0);
    for (const auto& ver1: vertices){
        for (const auto& ver2: vertices){
            double dis = distance(axis,
                                  ver1[0] - ver2[0],
                                  ver1[1] - ver2[1],
                                  ver1[2] - ver2[2]);
            if (max_length < dis) max_length = dis;
        }
    }

    int m = 30;
    double min_dis(1e10);
    vector1i vec_i{1,0,0}, vec_j{0,1,0}, vec_k{0,0,1};
    for (int i = -m; i < m + 1; ++i){
        double dis = distance(axis, i, 1, 0);
        if (min_dis > dis) {
            min_dis = dis;
            vec_j[0] = i;
        }
    }

    min_dis = 1e10;
    for (int i = -m; i < m + 1; ++i){
        for (int j = -m; j < m + 1; ++j){
            double dis = distance(axis, i, j, 1);
            if (min_dis > dis) {
                min_dis = dis;
                vec_k[0] = i;
                vec_k[1] = j;
            }
        }
    }

    m = 5;
    int i1, i2, i3;
    vector1i max_exp(3, 1);
    for (int i = -m; i < m + 1; ++i){
        for (int j = -m; j < m + 1; ++j){
            for (int k = -m; k < m + 1; ++k){
                i1 = vec_i[0] * i + vec_j[0] * j + vec_k[0] * k;
                i2 = vec_i[1] * i + vec_j[1] * j + vec_k[1] * k;
                i3 = vec_i[2] * i + vec_j[2] * j + vec_k[2] * k;
                double dis = distance(axis, i1, i2, i3);
                if (dis > 0 and dis < cutoff){
                    double exp = ceil(cutoff / dis);
                    if (exp * abs(i1) > max_exp[0]) max_exp[0] = exp * abs(i1);
                    if (exp * abs(i2) > max_exp[1]) max_exp[1] = exp * abs(i2);
                    if (exp * abs(i3) > max_exp[2]) max_exp[2] = exp * abs(i3);
                }
            }
        }
    }
    for (int l = 0; l < 3; ++l) max_exp[l] += 1;

    vector2d trans_c_array;
    for (int i = -max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = -max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = -max_exp[2]; k < max_exp[2] + 1; ++k){
                vector1d vec_c = prod(axis, vector1i{i,j,k});
                double dis = sqrt(vec_c[0]*vec_c[0]
                                +vec_c[1]*vec_c[1]
                                +vec_c[2]*vec_c[2]);
                if (dis < max_length + cutoff){
                    trans_c_array.emplace_back(vec_c);
                }
            }
        }
    }

    return trans_c_array;
}



const vector3d& Neighbor::get_dis_array() const{ return dis_array; }
const vector4d& Neighbor::get_diff_array() const{ return diff_array; }
const vector3i& Neighbor::get_atom2_array() const{ return atom2_array; }

