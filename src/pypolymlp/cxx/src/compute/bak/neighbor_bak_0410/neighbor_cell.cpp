/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_cell.h"


NeighborCell::NeighborCell(
    const vector2d& axis_i, const double cutoff_i)
    :axis(axis_i), cutoff(cutoff_i){

    axis0 = get_axis_vector(0);
    axis1 = get_axis_vector(1);
    axis2 = get_axis_vector(2);

    dot00 = dot_prod(axis0, axis0);
    dot11 = dot_prod(axis1, axis1);
    dot22 = dot_prod(axis2, axis2);
    dot01 = dot_prod(axis0, axis1);
    dot02 = dot_prod(axis0, axis2);
    dot12 = dot_prod(axis1, axis2);

    find_trans();
}

NeighborCell::~NeighborCell(){}


vector1d NeighborCell::get_axis_vector(const int col){
    return vector1d{axis[0][col], axis[1][col], axis[2][col]};
}

double NeighborCell::dot_prod(const vector1d& vec1, const vector1d& vec2){
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

double NeighborCell::norm(const vector1d& vec){
    return sqrt(dot_prod(vec, vec));
}

double NeighborCell::distance(const int i, const int j, const int k){

    double val1 = axis[0][0] * i + axis[0][1] * j + axis[0][2] * k;
    double val2 = axis[1][0] * i + axis[1][1] * j + axis[1][2] * k;
    double val3 = axis[2][0] * i + axis[2][1] * j + axis[2][2] * k;
    return sqrt(val1 * val1 + val2 * val2 + val3 * val3);
}

vector1d NeighborCell::to_cartesian(const int i, const int j, const int k){

    vector1d vec(3);
    vec[0] = axis[0][0] * i + axis[0][1] * j + axis[0][2] * k;
    vec[1] = axis[1][0] * i + axis[1][1] * j + axis[1][2] * k;
    vec[2] = axis[2][0] * i + axis[2][1] * j + axis[2][2] * k;
    return vec;
}


double NeighborCell::find_maximum_diagonal_in_cell(){

    vector2i vertices_comb = {{0,0,1}, {0,1,0}, {0,1,1},
                              {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1},
                              {0,0,-1}, {0,-1,0}, {0,-1,-1},
                              {-1,0,0}, {-1,0,-1}, {-1,-1,0}, {-1,-1,-1}};

    double max_length(0.0);
    for (const auto& ver: vertices_comb){
        double dis = distance(ver[0], ver[1], ver[2]);
        if (max_length < dis) max_length = dis;
    }

    return max_length;
}

double NeighborCell::optimize_1d_axis0(
    const int val1, const int val2, vector1i& vec){

    double imin = - (dot01 * val1 + dot02 * val2) / dot00;
    int i = round(imin);
    double min_dis = distance(i, val1, val2);
    vec = {i, val1, val2};
    return min_dis;
}


double NeighborCell::optimize_1d_axis1(
    const int val1, const int val2, vector1i& vec){

    double imin = - (dot01 * val1 + dot12 * val2) / dot11;
    int i = round(imin);
    double min_dis = distance(val1, i, val2);
    vec = {val1, i, val2};
    return min_dis;
}


double NeighborCell::optimize_1d_axis2(
    const int val1, const int val2, vector1i& vec){

    double imin = - (dot02 * val1 + dot12 * val2) / dot22;
    int i = round(imin);
    double min_dis = distance(val1, val2, i);
    vec = {val1, val2, i};
    return min_dis;
}


int NeighborCell::refine_single_axis(
    const bool ref01, const bool ref02, const bool ref12,
    vector1i& vec0, vector1i& vec1, vector1i& vec2){

    vec0 = {1,0,0}, vec1 = {0,1,0}, vec2 = {0,0,1};
    if (ref01 == true){
        if (dot00 > dot11){
            optimize_1d_axis1(1, 0, vec0); // vec[0] = 1, vec[2] = 0
        }
        else {
            optimize_1d_axis0(1, 0, vec1); // vec[1] = 1, vec[2] = 0
        }
    }
    else if (ref02 == true){
        if (dot00 > dot22){
            optimize_1d_axis2(1, 0, vec0); // vec[0] = 1, vec[1] = 0
        }
        else {
            optimize_1d_axis0(0, 1, vec2); // vec[1] = 0, vec[2] = 1
        }
    }
    else if (ref12 == true){
        if (dot11 > dot22){
            optimize_1d_axis2(0, 1, vec1); // vec[0] = 0, vec[1] = 1
        }
        else {
            optimize_1d_axis1(0, 1, vec2); // vec[0] = 0, vec[2] = 1
        }
    }

    return 0;
}


int NeighborCell::refine_axis(vector1i& vec0, vector1i& vec1, vector1i& vec2){

    int i, j;
    double imin, jmin, denom;

    denom = dot11 * dot22 - dot12 * dot12;
    imin = (dot02 * dot12 - dot01 * dot22) / denom;
    jmin = (dot01 * dot12 - dot02 * dot11) / denom;
    i = round(imin), j = round(jmin);
    vec0 = {1, i, j};

    denom = dot00 * dot22 - dot02 * dot02;
    imin = (dot02 * dot12 - dot01 * dot22) / denom;
    jmin = (dot01 * dot02 - dot12 * dot00) / denom;
    i = round(imin), j = round(jmin);
    vec1 = {i, 1, j};

    denom = dot00 * dot11 - dot01 * dot01;
    imin = (dot01 * dot12 - dot02 * dot11) / denom;
    jmin = (dot01 * dot02 - dot12 * dot00) / denom;
    i = round(imin), j = round(jmin);
    vec2 = {i, j, 1};

    if (vec0[0] == vec1[0] and vec0[1] == vec1[1] and vec0[2] == vec1[2]) 
        vec0[1] = 0;
    if (vec0[0] == vec2[0] and vec0[1] == vec2[1] and vec0[2] == vec2[2]) 
        vec0[2] = 0;
    if (vec1[0] == vec2[0] and vec1[1] == vec2[1] and vec1[2] == vec2[2]) 
        vec1[2] = 0;

    return 0;
}


int NeighborCell::find_trans(){

    bool ref01(false), ref02(false), ref12(false);
    if (abs(dot01) > 0.5 * dot00 or abs(dot01) > 0.5 * dot11) ref01 = true;
    if (abs(dot02) > 0.5 * dot00 or abs(dot02) > 0.5 * dot22) ref02 = true;
    if (abs(dot12) > 0.5 * dot11 or abs(dot12) > 0.5 * dot22) ref12 = true;
    const auto ref_sum = ref01 + ref02 + ref12;

    vector1i vec0, vec1, vec2;
    if (ref_sum == 0){
        vec0 = {1,0,0}, vec1 = {0,1,0}, vec2 = {0,0,1};
    }
    else if (ref_sum == 1){
        refine_single_axis(ref01, ref02, ref12, vec0, vec1, vec2);
    }
    else refine_axis(vec0, vec1, vec2);
    
    double dis;
    vector1i max_exp(3, 1);
    vector2i vecs = {vec0, vec1, vec2};
    for (const auto& v: vecs){
        dis = distance(v[0], v[1], v[2]);
        int exp = ceil(cutoff / dis);
        if (exp * abs(v[0]) > max_exp[0]) max_exp[0] = exp * abs(v[0]);
        if (exp * abs(v[1]) > max_exp[1]) max_exp[1] = exp * abs(v[1]);
        if (exp * abs(v[2]) > max_exp[2]) max_exp[2] = exp * abs(v[2]);
    }
    for (int i = 0; i < 3; ++i) max_exp[i] += 1;

    // Must check how to screen required translations
    double max_length = find_maximum_diagonal_in_cell();
    vector1d vec_c;
    for (int i = - max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = - max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = - max_exp[2]; k < max_exp[2] + 1; ++k){
                vec_c = to_cartesian(i,j,k);
                dis = norm(vec_c);
                if (dis < 2 * max_length + cutoff){
                    trans_c_array.emplace_back(vec_c);
                }
            }
        }
    }

    return 0;
}


const vector2d& NeighborCell::get_translations() const{ return trans_c_array; }


/*
int NeighborCell::refine_axis(const vector2d& axis, 
                              vector1i& vec0, vector1i& vec1, vector1i& vec2){

    int istep(10);
    int imax(istep), imin(0);
    double min_dis0(1e10), min_dis1(1e10), min_dis2(1e10), dis;
    vector1d axis0, axis1, axis2;
    double dot00, dot11, dot22, dot01, dot02, dot12;
    bool repeat = true;

    while (imax < 51 and repeat == true){
        vector1i icand;
        for (int i = -imax + 1; i < -imin + 1; ++i){
            icand.emplace_back(i);
        }
        for (int i = imin; i < imax; ++i){
            if (i != 0) icand.emplace_back(i);
        }

        for (const auto i: icand){
            vector1i vec0_trial;
            dis = optimize_1d_axis2(axis, 1, i, vec0_trial); 
            if (min_dis0 > dis) {
                min_dis0 = dis;
                vec0 = vec0_trial;
            }
        }

        for (const auto i: icand){
            if (i != 1){
                vector1i vec1_trial;
                dis = optimize_1d_axis2(axis, i, 1, vec1_trial);
                if (min_dis1 > dis) {
                    min_dis1 = dis;
                    vec1 = vec1_trial;
                }
            }
        }

        for (const auto i: icand){
            if (i != 1){
                vector1i vec2_trial;
                dis = optimize_1d_axis1(axis, i, 1, vec2_trial); 
                if (min_dis2 > dis) {
                    min_dis2 = dis;
                    vec2 = vec2_trial;
                }
            }
        }
        if (vec2[1] == 1) vec2[1] = 0;

        axis0 = dot_prod(axis, vec0);
        axis1 = dot_prod(axis, vec1);
        axis2 = dot_prod(axis, vec1);
        dot00 = dot_prod(axis0, axis0);
        dot11 = dot_prod(axis1, axis1);
        dot22 = dot_prod(axis2, axis2);
        dot01 = dot_prod(axis0, axis1);
        dot02 = dot_prod(axis0, axis2);
        dot12 = dot_prod(axis1, axis2);
        if (abs(dot01) <= 0.5 * dot00 and abs(dot01) <= 0.5 * dot11 and
            abs(dot02) <= 0.5 * dot00 and abs(dot02) <= 0.5 * dot22 and
            abs(dot12) <= 0.5 * dot11 and abs(dot12) <= 0.5 * dot22){
            repeat = false;
        }
        imax += istep;
        imin += istep;
    }
    return 0;
}
*/

/*    int m = 1;
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
    */

/*
    int m = 1, i1, i2, i3;
    double dis;
    vector1i max_exp(3, 1);
    for (int i = 0; i < m + 1; ++i){
        for (int j = 0; j < m + 1; ++j){
            for (int k = 0; k < m + 1; ++k){
                i1 = vec0[0] * i + vec1[0] * j + vec2[0] * k;
                i2 = vec0[1] * i + vec1[1] * j + vec2[1] * k;
                i3 = vec0[2] * i + vec1[2] * j + vec2[2] * k;
                dis = distance(i1, i2, i3);
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
*/


