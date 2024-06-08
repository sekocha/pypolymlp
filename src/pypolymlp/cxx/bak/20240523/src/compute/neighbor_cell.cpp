/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_cell.h"


NeighborCell::NeighborCell(
    const vector2d& axis_i,
    const vector2d& positions_c_i,
    const double cutoff_i
):axis(axis_i), positions_c(positions_c_i), cutoff(cutoff_i){

    compute_metric();
    find_trans();
}

NeighborCell::~NeighborCell(){}


int NeighborCell::compute_metric(){

    axis0 = get_axis_vector(0);
    axis1 = get_axis_vector(1);
    axis2 = get_axis_vector(2);

    dot00 = dot_prod(axis0, axis0);
    dot11 = dot_prod(axis1, axis1);
    dot22 = dot_prod(axis2, axis2);
    dot01 = dot_prod(axis0, axis1);
    dot02 = dot_prod(axis0, axis2);
    dot12 = dot_prod(axis1, axis2);

    ref01 = false, ref02 = false, ref12 = false;
    if (abs(dot01) > 0.5 * dot00 or abs(dot01) > 0.5 * dot11) ref01 = true;
    if (abs(dot02) > 0.5 * dot00 or abs(dot02) > 0.5 * dot22) ref02 = true;
    if (abs(dot12) > 0.5 * dot11 or abs(dot12) > 0.5 * dot22) ref12 = true;
    ref_sum = ref01 + ref02 + ref12;

    return 0;
}

vector1d NeighborCell::get_axis_vector(const int col){
    return vector1d{axis[0][col], axis[1][col], axis[2][col]};
}


int NeighborCell::replace_axis_vector(const vector1i& vec, const int col){

    const vector1d& a0 = to_cartesian(vec[0], vec[1], vec[2]);
    for (int i = 0; i < 3; ++i) axis[i][col] = a0[i];
    compute_metric();

    return 0;
}


vector1d NeighborCell::dot_prod(const vector2d& mat, const vector1d& vec2){

    vector1d res(mat.size());
    int idx(0);
    for (const auto& vec1: mat) {
        res[idx] = dot_prod(vec1, vec2);
        ++idx;
    }
    return res;
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


int NeighborCell::refine_axis(){

    vector1i vec_lc;
    int iter(0);
    while (ref_sum > 0 and iter < 100){
        if (ref01 == true){
            if (dot00 > dot11) {
                optimize_1d_axis1(1, 0, vec_lc); // vec = [1,i,0]
                replace_axis_vector(vec_lc, 0);
            }
            else {
                optimize_1d_axis0(1, 0, vec_lc); // vec = [i,1,0]
                replace_axis_vector(vec_lc, 1);
            }

        }

        if (ref02 == true){
            if (dot00 > dot22) {
                optimize_1d_axis2(1, 0, vec_lc); // vec = [1,0,i]
                replace_axis_vector(vec_lc, 0);
            }
            else {
                optimize_1d_axis0(0, 1, vec_lc); // vec = [i,0,1]
                replace_axis_vector(vec_lc, 2);
            }
        }

        if (ref12 == true){
            if (dot11 > dot22) {
                optimize_1d_axis2(0, 1, vec_lc); // vec = [0,1,i]
                replace_axis_vector(vec_lc, 1);
            }
            else {
                optimize_1d_axis1(0, 1, vec_lc); // vec = [0,i,1]
                replace_axis_vector(vec_lc, 2);
            }
        }
        ++iter;
    }

    return 0;
}

int NeighborCell::calc_inverse_axis(){

    axis_inv = vector2d(3, vector1d(3));
    double det = axis[0][0] * axis[1][1] * axis[2][2]
               + axis[0][1] * axis[1][2] * axis[2][0]
               + axis[0][2] * axis[1][0] * axis[2][1]
               - axis[0][2] * axis[1][1] * axis[2][0]
               - axis[0][1] * axis[1][0] * axis[2][2]
               - axis[0][0] * axis[1][2] * axis[2][1];
    axis_inv[0][0] = axis[1][1] * axis[2][2] - axis[1][2] * axis[2][1];
    axis_inv[0][1] = - (axis[0][1] * axis[2][2] - axis[0][2] * axis[2][1]);
    axis_inv[0][2] = axis[0][1] * axis[1][2] - axis[0][2] * axis[1][1];
    axis_inv[1][0] = - (axis[1][0] * axis[2][2] - axis[1][2] * axis[2][0]);
    axis_inv[1][1] = axis[0][0] * axis[2][2] - axis[0][2] * axis[2][0];
    axis_inv[1][2] = - (axis[0][0] * axis[1][2] - axis[0][2] * axis[1][0]);
    axis_inv[2][0] = axis[1][0] * axis[2][1] - axis[1][1] * axis[2][0];
    axis_inv[2][1] = - (axis[0][0] * axis[2][1] - axis[0][1] * axis[2][0]);
    axis_inv[2][2] = axis[0][0] * axis[1][1] - axis[0][1] * axis[1][0];
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j) axis_inv[i][j] /= det;
    }
    return 0;
}

int NeighborCell::standardize_positions(vector2d& positions){

    for (auto& pos: positions){
        for (auto& p: pos) p -= floor(p);
    }
    return 0;
}


int NeighborCell::find_trans(){

    if (ref_sum > 0){
        refine_axis();
        calc_inverse_axis();
        /*
            for (int i = 0; i < 3; ++i){
                for (int j = 0; j < 3; ++j){
                    std::cout << axis[i][j] << ", ";
                }
                std::cout << std::endl;
            }
        */
        int n_atom = positions_c[0].size();
        vector2d positions(3, vector1d(n_atom));
        vector1d pos_c(3), pos(3);
        for (int j = 0; j < n_atom; ++j){
            pos_c[0] = positions_c[0][j];
            pos_c[1] = positions_c[1][j];
            pos_c[2] = positions_c[2][j];
            pos = dot_prod(axis_inv, pos_c);
            positions[0][j] = pos[0];
            positions[1][j] = pos[1];
            positions[2][j] = pos[2];
        }
        standardize_positions(positions);
        for (int j = 0; j < n_atom; ++j){
            pos_c = to_cartesian(
                positions[0][j], positions[1][j], positions[2][j]);
            positions_c[0][j] = pos_c[0];
            positions_c[1][j] = pos_c[1];
            positions_c[2][j] = pos_c[2];
        }
    }

    vector1i max_exp = {int(ceil(cutoff / distance(1, 0, 0)) + 1),
                        int(ceil(cutoff / distance(0, 1, 0)) + 1),
                        int(ceil(cutoff / distance(0, 0, 1)) + 1)};

    double dis;
    double max_length = find_maximum_diagonal_in_cell();
    vector1d vec_c;
    for (int i = - max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = - max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = - max_exp[2]; k < max_exp[2] + 1; ++k){
                vec_c = to_cartesian(i, j, k);
                dis = norm(vec_c);
                if (dis < max_length + cutoff){
                    trans_c_array.emplace_back(vec_c);
                }
            }
        }
    }

    return 0;
}


const vector2d& NeighborCell::get_axis() const { return axis; }
const vector2d& NeighborCell::get_positions_cartesian() const {
    return positions_c;
}
const vector2d& NeighborCell::get_translations() const { return trans_c_array; }
