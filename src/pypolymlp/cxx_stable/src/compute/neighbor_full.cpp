/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_full.h"

NeighborFull::NeighborFull(
    const vector2d& axis,
    const vector2d& positions_c,
    const double cutoff
){

    NeighborCell neigh_cell(axis, positions_c, cutoff);
    const auto& trans = neigh_cell.get_translations();
    const auto& positions_c_rev = neigh_cell.get_positions_cartesian();

    n_total_atom = positions_c[0].size();
    const double tol = 1e-10;
    const double tol_sq = tol * tol;
    const double cutoff_sq = cutoff * cutoff;

    vector2i neighbor_atoms(n_total_atom);
    vector2d dx_tmp(n_total_atom);
    vector2d dy_tmp(n_total_atom);
    vector2d dz_tmp(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        auto& jlocal = neighbor_atoms[i];
        auto& dxlocal = dx_tmp[i];
        auto& dylocal = dy_tmp[i];
        auto& dzlocal = dz_tmp[i];
        for (int j = 0; j < n_total_atom; ++j){
            double dx_ij = positions_c_rev[0][j] - positions_c_rev[0][i];
            double dy_ij = positions_c_rev[1][j] - positions_c_rev[1][i];
            double dz_ij = positions_c_rev[2][j] - positions_c_rev[2][i];
            for (const auto& tr: trans){
                double dx = dx_ij + tr[0];
                double dy = dy_ij + tr[1];
                double dz = dz_ij + tr[2];
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < cutoff_sq and r2 > tol_sq) {
                    jlocal.emplace_back(j);
                    dxlocal.emplace_back(dx);
                    dylocal.emplace_back(dy);
                    dzlocal.emplace_back(dz);
                }
            }
        }
    }

    offset.resize(n_total_atom + 1);
    offset[0] = 0;
    for (int i = 0; i < n_total_atom; ++i){
        int count = neighbor_atoms[i].size();
        offset[i+1] = offset[i] + count;
    }

    int N = offset[n_total_atom];
    neigh.resize(N);
    dx.resize(N);
    dy.resize(N);
    dz.resize(N);

    int id = 0;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < neighbor_atoms[i].size(); ++j){
            neigh[id] = neighbor_atoms[i][j];
            dx[id] = dx_tmp[i][j];
            dy[id] = dy_tmp[i][j];
            dz[id] = dz_tmp[i][j];
            ++id;
        }
    }
}

NeighborFull::~NeighborFull(){}


// For test
vector3d NeighborFull::get_dis_array(const int n_type, const vector1i& types){
    vector3d dis_array(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        dis_array[i].resize(n_type);
        for (int k = offset[i]; k < offset[i+1]; ++k){
            int j = neigh[k];
            double x = dx[k];
            double y = dy[k];
            double z = dz[k];
            double dis = sqrt(x*x + y*y + z*z);
            int jtype = types[j];
            dis_array[i][jtype].emplace_back(dis);
        }
    }
    return dis_array;
}
// For test
vector4d NeighborFull::get_diff_array(const int n_type, const vector1i& types){
    vector4d diff_array(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        diff_array[i].resize(n_type);
        for (int k = offset[i]; k < offset[i+1]; ++k){
            int j = neigh[k];
            double x = dx[k];
            double y = dy[k];
            double z = dz[k];
            int jtype = types[j];
            diff_array[i][jtype].emplace_back(vector1d{x, y, z});
        }
    }
    return diff_array;
}
// For test
vector3i NeighborFull::get_atom2_array(const int n_type, const vector1i& types){
    vector3i atom2_array(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        atom2_array[i].resize(n_type);
        for (int k = offset[i]; k < offset[i+1]; ++k){
            int j = neigh[k];
            int jtype = types[j];
            atom2_array[i][jtype].emplace_back(j);
        }
    }
    return atom2_array;
}
