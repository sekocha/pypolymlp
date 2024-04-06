/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_cell.h"


NeighborCell::NeighborCell(const vector2d& axis, const double cutoff){

    const auto& trans = find_trans(axis, cutoff);
}

NeighborCell::~NeighborCell(){}


vector1d NeighborCell::dot_prod(const vector2d& mat, const vector1i& vec){

    vector1d res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i){
        for (size_t j = 0; j < mat[i].size(); ++j){
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}

double NeighborCell::dot_prod(const vector1d& vec1, const vector1d& vec2){

    double res(0.0);
    for (size_t i = 0; i < vec1.size(); ++i) res += vec1[i] * vec2[i];
    return res;
}

double NeighborCell::norm(const vector1d& vec){
    return sqrt(dot_prod(vec,vec));
}

vector1d NeighborCell::get_axis_vector(const vector2d& axis, const int col){
    return vector1d{axis[0][col], axis[1][col], axis[2][col]};
}



double NeighborCell::distance(const vector2d& axis, 
                              const int i, const int j, const int k){
    vector1d vec_c = dot_prod(axis, vector1i{i,j,k});
    return sqrt(dot_prod(vec_c, vec_c));
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

int NeighborCell::refine_single_axis(
    const vector2d& axis, 
    const bool ref01, const bool ref02, const bool ref12,
    const double dot00, const double dot11, const double dot22,
    const double dot01, const double dot02, const double dot12,
    vector1i& vec0, vector1i& vec1, vector1i& vec2){

    double min_dis(1e10), dis;
    vec0 = {1,0,0}, vec1 = {0,1,0}, vec2 = {0,0,1};

    const int m(10);
    if (ref01 == true and dot00 > dot11){
        for (int i = -m; i < m + 1; ++i){
            dis = distance(axis, 1, i, 0);
            if (min_dis > dis) {
                min_dis = dis;
                vec0 = {1, i, 0};
            }
            else break;
        }
    }
    else if (ref01 == true and dot11 >= dot00){
        for (int i = -m; i < m + 1; ++i){
            dis = distance(axis, i, 1, 0);
            if (min_dis > dis) {
                min_dis = dis;
                vec1 = {i, 1, 0};
            }
            else break;
        }
    }
    else if (ref02 == true and dot00 > dot22){
        for (int i = -m; i < m + 1; ++i){
            dis = distance(axis, 1, 0, i);
            if (min_dis > dis) {
                min_dis = dis;
                vec0 = {1, 0, i};
            }
            else break;
        }
    }
    else if (ref02 == true and dot22 >= dot00){
        for (int i = -m; i < m + 1; ++i){
            dis = distance(axis, i, 0, 1);
            if (min_dis > dis) {
                min_dis = dis;
                vec2 = {i, 0, 1};
            }
            else break;
        }
    }
    else if (ref12 == true and dot11 > dot22){
        for (int i = -m; i < m + 1; ++i){
            dis = distance(axis, 0, 1, i);
            if (min_dis > dis) {
                min_dis = dis;
                vec1 = {0, 1, i};
            }
            else break;
        }
    }
    else if (ref12 == true and dot22 >= dot11){
        for (int i = -m; i < m + 1; ++i){
            dis = distance(axis, 0, i, 1);
            if (min_dis > dis) {
                min_dis = dis;
                vec2 = {0, i, 1};
            }
            else break;
        }
    }

    return 0;
}

int NeighborCell::refine_axis(const vector2d& axis, 
                              vector1i& vec0, vector1i& vec1, vector1i& vec2){

    const int m(50);
    double min_dis(1e10), dis;
    for (int i = -m; i < m + 1; ++i){
        double local(1e10);
        for (int j = -m; j < m + 1; ++j){
            dis = distance(axis, 1, i, j);
            if (min_dis > dis) {
                min_dis = dis;
                vec0 = {1, i, j};
            }
            if (local > dis) local = dis;
            else break;
        }
    }

    min_dis = 1e10;
    for (int i = -m; i < m + 1; ++i){
        if (i != 1){
            double local(1e10);
            for (int j = -m; j < m + 1; ++j){
                dis = distance(axis, i, 1, j);
                if (min_dis > dis) {
                    min_dis = dis;
                    vec1 = {i, 1, j};
                }
                if (local > dis) local = dis;
                else break;
            }
        }
    }

    min_dis = 1e10;
    for (int i = -m; i < m + 1; ++i){
        if (i != 1){
            double local(1e10);
            for (int j = -m; j < m + 1; ++j){
                if (j != 1){
                    dis = distance(axis, i, j, 1);
                    if (min_dis > dis) {
                        min_dis = dis;
                        vec2 = {i, j, 1};
                    }
                    if (local > dis) local = dis;
                    else break;
                }
            }
        }
    }

    return 0;
}


int NeighborCell::find_trans(const vector2d& axis, const double cutoff){

    vector1d axis0 = get_axis_vector(axis, 0);
    vector1d axis1 = get_axis_vector(axis, 1);
    vector1d axis2 = get_axis_vector(axis, 2);

    double dot00 = dot_prod(axis0, axis0);
    double dot11 = dot_prod(axis1, axis1);
    double dot22 = dot_prod(axis2, axis2);
    double dot01 = dot_prod(axis0, axis1);
    double dot02 = dot_prod(axis0, axis2);
    double dot12 = dot_prod(axis1, axis2);

    bool ref01(false), ref02(false), ref12(false);
    if (abs(dot01) > 0.5 * dot00 or abs(dot01) > 0.5 * dot11) ref01 = true;
    if (abs(dot02) > 0.5 * dot00 or abs(dot02) > 0.5 * dot22) ref02 = true;
    if (abs(dot12) > 0.5 * dot11 or abs(dot12) > 0.5 * dot22) ref12 = true;
    const auto ref_sum = ref01 + ref02 + ref12;

    vector1i vec0, vec1, vec2;
    if (ref_sum == 0){
        vec0 = {1,0,0};
        vec1 = {0,1,0};
        vec2 = {0,0,1};
    }
    else if (ref_sum == 1){
        refine_single_axis(axis, ref01, ref02, ref12,
                           dot00, dot11, dot22, dot01, dot02, dot12,
                           vec0, vec1, vec2);
    }
    else if (ref_sum > 1){
        refine_axis(axis, vec0, vec1, vec2);
    }

    int m = 5, i1, i2, i3;
    double dis;
    vector1i max_exp(3, 1);
    for (int i = -m; i < m + 1; ++i){
        for (int j = -m; j < m + 1; ++j){
            for (int k = -m; k < m + 1; ++k){
                i1 = vec0[0] * i + vec1[0] * j + vec2[0] * k;
                i2 = vec0[1] * i + vec1[1] * j + vec2[1] * k;
                i3 = vec0[2] * i + vec1[2] * j + vec2[2] * k;
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

    double max_length = find_maximum_diagonal_in_cell(axis);

    for (int i = -max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = -max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = -max_exp[2]; k < max_exp[2] + 1; ++k){
                vector1d vec_c = dot_prod(axis, vector1i{i,j,k});
                double dis = norm(vec_c);
                if (dis < max_length + cutoff){
                    trans_c_array.emplace_back(vec_c);
                }
            }
        }
    }

    return 0;
}


const vector2d& NeighborCell::get_translations() const{ return trans_c_array; }


