/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_cell.h"


NeighborCell::NeighborCell(const vector2d& axis, const double cutoff){

    const auto& trans = find_trans(axis, cutoff);
}

NeighborCell::~NeighborCell(){}


vector1d NeighborCell::prod(const vector2d& mat, const vector1i& vec){

    vector1d res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i){
        for (size_t j = 0; j < mat[i].size(); ++j){
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}


double NeighborCell::distance(const vector2d& axis, 
                              const int i, const int j, const int k){
    vector1d vec_c = prod(axis, vector1i{i,j,k});
    return sqrt(vec_c[0]*vec_c[0]+vec_c[1]*vec_c[1]+vec_c[2]*vec_c[2]);
}


double NeighborCell::norm(const vector1d& vec){
    return sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
}


double NeighborCell::find_maximum_diagonal_in_cell(const vector2d& axis){

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
            double dis = distance(
                axis, ver1[0] - ver2[0], ver1[1] - ver2[1], ver1[2] - ver2[2]
            );
            if (max_length < dis) max_length = dis;
        }
    }

    return max_length;
}


int NeighborCell::find_trans(const vector2d& axis, const double cutoff){

    double max_length = find_maximum_diagonal_in_cell(axis);

    double min_dis, dis;
    vector1i vec_i{1,0,0}, vec_j{0,1,0}, vec_k{0,0,1};

    double norm_i = norm(vector1d{axis[0][0], axis[1][0], axis[2][0]});
    double norm_j = norm(vector1d{axis[0][1], axis[1][1], axis[2][1]});
    double norm_k = norm(vector1d{axis[0][2], axis[1][2], axis[2][2]});

    int m = 30;
    if (norm_i > norm_j or norm_i > norm_k){
        min_dis = 1e10;
        for (int i = -m; i < m + 1; ++i){
            double local(1e10);
            for (int j = -m; j < m + 1; ++j){
                dis = distance(axis, 1, i, j);
                if (min_dis > dis) {
                    min_dis = dis;
                    vec_i[1] = i;
                    vec_i[2] = j;
                }
                if (local > dis) local = dis;
                else break;
            }
        }
    }

    if (norm_j >= norm_i or norm_j > norm_k){
        min_dis = 1e10;
        for (int i = -m; i < m + 1; ++i){
            double local(1e10);
            for (int j = -m; j < m + 1; ++j){
                dis = distance(axis, i, 1, j);
                if (min_dis > dis) {
                    min_dis = dis;
                    vec_j[0] = i;
                    vec_j[2] = j;
                }
                if (local > dis) local = dis;
                else break;
            }
        }
    }

    if (norm_k >= norm_i or norm_k >= norm_j){
        min_dis = 1e10;
        for (int i = -m; i < m + 1; ++i){
            double local(1e10);
            for (int j = -m; j < m + 1; ++j){
                dis = distance(axis, i, j, 1);
                if (min_dis > dis) {
                    min_dis = dis;
                    vec_k[0] = i;
                    vec_k[1] = j;
                }
                if (local > dis) local = dis;
                else break;
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
                dis = distance(axis, i1, i2, i3);
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

    for (int i = -max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = -max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = -max_exp[2]; k < max_exp[2] + 1; ++k){
                vector1d vec_c = prod(axis, vector1i{i,j,k});
                double dis = sqrt(
                    vec_c[0]*vec_c[0] + vec_c[1]*vec_c[1] + vec_c[2]*vec_c[2]
                );
                if (dis < max_length + cutoff){
                    trans_c_array.emplace_back(vec_c);
                }
            }
        }
    }

    return 0;
}


const vector2d& NeighborCell::get_translations() const{ return trans_c_array; }


