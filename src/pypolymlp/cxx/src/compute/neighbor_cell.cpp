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

double NeighborCell::optimize_1d_axis0(
    const vector2d& axis, const int val1, const int val2,
    vector1i& vec){

    double min_dis(1e10), dis(0.0), dis0, dis1;
    int i, increment;
    dis0 = distance(axis, 0, val1, val2); 
    dis1 = distance(axis, 1, val1, val2);
    if (dis0 < dis1) {
        i = 0, increment = -1;
        dis = dis0;
    }
    else {
        i = 1, increment = 1;
        dis = dis1;
    }
    while (min_dis > dis){
        min_dis = dis;
        vec = {i, val1, val2};
        i += increment;
        dis = distance(axis, i, val1, val2);
    }
    return min_dis;
}


double NeighborCell::optimize_1d_axis1(
    const vector2d& axis, const int val1, const int val2,
    vector1i& vec){

    double min_dis(1e10), dis(0.0), dis0, dis1;
    int i, increment;
    dis0 = distance(axis, val1, 0, val2);
    dis1 = distance(axis, val1, 1, val2);
    if (dis0 < dis1) {
        i = 0, increment = -1;
        dis = dis0;
    }
    else {
        i = 1, increment = 1;
        dis = dis1;
    }
    while (min_dis > dis){
        min_dis = dis;
        vec = {val1, i, val2};
        i += increment;
        dis = distance(axis, val1, i, val2);
    }
    return min_dis;
}


double NeighborCell::optimize_1d_axis2(
    const vector2d& axis, const int val1, const int val2,
    vector1i& vec){

    double min_dis(1e10), dis(0.0), dis0, dis1;
    int i, increment;
    dis0 = distance(axis, val1, val2, 0); 
    dis1 = distance(axis, val1, val2, 1);
    if (dis0 < dis1) {
        i = 0, increment = -1;
        dis = dis0;
    }
    else {
        i = 1, increment = 1;
        dis = dis1;
    }
    while (min_dis > dis){
        min_dis = dis;
        vec = {val1, val2, i};
        i += increment;
        dis = distance(axis, val1, val2, i);
    }
    return min_dis;
}


int NeighborCell::refine_single_axis(
    const vector2d& axis, 
    const bool ref01, const bool ref02, const bool ref12,
    const double dot00, const double dot11, const double dot22,
    const double dot01, const double dot02, const double dot12,
    vector1i& vec0, vector1i& vec1, vector1i& vec2){

    vec0 = {1,0,0}, vec1 = {0,1,0}, vec2 = {0,0,1};
    if (ref01 == true and dot00 > dot11){
        optimize_1d_axis1(axis, 1, 0, vec0); // vec[0] = 1, vec[2] = 0
    }
    else if (ref01 == true and dot11 >= dot00){
        optimize_1d_axis0(axis, 1, 0, vec1); // vec[1] = 1, vec[2] = 0
    }
    else if (ref02 == true and dot00 > dot22){
        optimize_1d_axis2(axis, 1, 0, vec0); // vec[0] = 1, vec[1] = 0
    }
    else if (ref02 == true and dot22 >= dot00){
        optimize_1d_axis0(axis, 0, 1, vec2); // vec[1] = 0, vec[2] = 1
    }
    else if (ref12 == true and dot11 > dot22){
        optimize_1d_axis2(axis, 0, 1, vec1); // vec[0] = 0, vec[1] = 1
    }
    else if (ref12 == true and dot22 >= dot11){
        optimize_1d_axis1(axis, 0, 1, vec2); // vec[0] = 0, vec[2] = 1
    }
    return 0;
}

int NeighborCell::refine_axis(const vector2d& axis, 
                              const double dot00, 
                              const double dot11, 
                              const double dot22, 
                              const double dot01, 
                              const double dot02, 
                              const double dot12, 
                              vector1i& vec0, vector1i& vec1, vector1i& vec2){

    int m = 1;
    int ratio1 = ceil(abs(dot01 / dot00));
    int ratio2 = ceil(abs(dot01 / dot11));
    int ratio3 = ceil(abs(dot02 / dot00));
    int ratio4 = ceil(abs(dot02 / dot22));
    int ratio5 = ceil(abs(dot12 / dot11));
    int ratio6 = ceil(abs(dot12 / dot22));
    if (ratio1 != 0) m *= ratio1;
    if (ratio2 != 0) m *= ratio2;
    if (ratio3 != 0) m *= ratio3;
    if (ratio4 != 0) m *= ratio4;
    if (ratio5 != 0) m *= ratio5;
    if (ratio6 != 0) m *= ratio6;
    m += 10;

    double min_dis(1e10), dis;
    for (int i = -m; i < m + 1; ++i){
        vector1i vec0_trial;
        dis = optimize_1d_axis2(axis, 1, i, vec0_trial); //vec[0,1] = 1,i
        if (min_dis > dis) {
            min_dis = dis;
            vec0 = vec0_trial;
        }
    }

    min_dis = 1e10;
    for (int i = -m; i < m + 1; ++i){
        if (i != 1){
            vector1i vec1_trial;
            dis = optimize_1d_axis2(axis, i, 1, vec1_trial); //vec[0,1] = i, 1
            if (min_dis > dis) {
                min_dis = dis;
                vec1 = vec1_trial;
            }
        }
    }

    min_dis = 1e10;
    for (int i = -m; i < m + 1; ++i){
        if (i != 1){
            vector1i vec2_trial;
            dis = optimize_1d_axis1(axis, i, 1, vec2_trial); //vec[0,2] = i,1
            if (min_dis > dis) {
                min_dis = dis;
                vec2 = vec2_trial;
            }
        }
    }
    if (vec2[1] == 1) vec2[1] = 0;

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
    else refine_axis(axis, dot00, dot11, dot22, dot01, dot02, dot12, 
                     vec0, vec1, vec2);

    int m = 3, i1, i2, i3;
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


//int NeighborCell::refine_axis(const vector2d& axis, 
//                              vector1i& vec0, vector1i& vec1, vector1i& vec2){
//
//    int istep(10);
//    int imax(istep), imin(0);
//    double min_dis0(1e10), min_dis1(1e10), min_dis2(1e10), dis;
//    vector1d axis0, axis1, axis2;
//    double dot00, dot11, dot22, dot01, dot02, dot12;
//    bool repeat = true;
//
//    while (imax < 51 and repeat == true){
//        vector1i icand;
//        for (int i = -imax + 1; i < -imin + 1; ++i){
//            icand.emplace_back(i);
//        }
//        for (int i = imin; i < imax; ++i){
//            if (i != 0) icand.emplace_back(i);
//        }
//
//        for (const auto i: icand){
//            vector1i vec0_trial;
//            dis = optimize_1d_axis2(axis, 1, i, vec0_trial); 
//            if (min_dis0 > dis) {
//                min_dis0 = dis;
//                vec0 = vec0_trial;
//            }
//        }
//
//        for (const auto i: icand){
//            if (i != 1){
//                vector1i vec1_trial;
//                dis = optimize_1d_axis2(axis, i, 1, vec1_trial);
//                if (min_dis1 > dis) {
//                    min_dis1 = dis;
//                    vec1 = vec1_trial;
//                }
//            }
//        }
//
//        for (const auto i: icand){
//            if (i != 1){
//                vector1i vec2_trial;
//                dis = optimize_1d_axis1(axis, i, 1, vec2_trial); 
//                if (min_dis2 > dis) {
//                    min_dis2 = dis;
//                    vec2 = vec2_trial;
//                }
//            }
//        }
//        if (vec2[1] == 1) vec2[1] = 0;
//
///*
//        axis0 = dot_prod(axis, vec0);
//        axis1 = dot_prod(axis, vec1);
//        axis2 = dot_prod(axis, vec1);
//        dot00 = dot_prod(axis0, axis0);
//        dot11 = dot_prod(axis1, axis1);
//        dot22 = dot_prod(axis2, axis2);
//        dot01 = dot_prod(axis0, axis1);
//        dot02 = dot_prod(axis0, axis2);
//        dot12 = dot_prod(axis1, axis2);
//        if (abs(dot01) <= 0.5 * dot00 and abs(dot01) <= 0.5 * dot11 and
//            abs(dot02) <= 0.5 * dot00 and abs(dot02) <= 0.5 * dot22 and
//            abs(dot12) <= 0.5 * dot11 and abs(dot12) <= 0.5 * dot22){
//            repeat = false;
//        }
//        */
//        imax += istep;
//        imin += istep;
//    }
//    std::cout << vec0[0] << " " << vec0[1] << " " << vec0[2] << std::endl;
//    std::cout << vec1[0] << " " << vec1[1] << " " << vec1[2] << std::endl;
//    std::cout << vec2[0] << " " << vec2[1] << " " << vec2[2] << std::endl;
//
//    return 0;
//}



