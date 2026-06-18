/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_half.h"

NeighborHalf::NeighborHalf(
    const vector2d& axis,
    const vector2d& positions_c,
    const double cutoff,
    const bool use_openmp
){

    int num_threads = omp_get_max_threads();
    if (use_openmp) omp_set_num_threads(num_threads);
    else omp_set_num_threads(1);

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
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int i = 0; i < n_total_atom; ++i){
        auto& jlocal = neighbor_atoms[i];
        auto& dxlocal = dx_tmp[i];
        auto& dylocal = dy_tmp[i];
        auto& dzlocal = dz_tmp[i];
        for (int j = 0; j < i; ++j){
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
        for (const auto& tr: trans){
            double dx = tr[0];
            double dy = tr[1];
            double dz = tr[2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < cutoff_sq and r2 > tol_sq){
                bool keep;
                if (dz >= tol) keep = true;
                else if (fabs(dz) < tol and dy >= tol) keep = true;
                else if (fabs(dz) < tol and fabs(dy) < tol and dx >= tol)
                    keep = true;
                else keep = false;
                if (keep){
                    jlocal.emplace_back(i);
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

    omp_set_num_threads(num_threads);
}

NeighborHalf::~NeighborHalf(){}

// For test
vector2i NeighborHalf::get_half_list(){
    vector2i half_list(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        for (int k = offset[i]; k < offset[i+1]; ++k){
            half_list[i].emplace_back(neigh[k]);
        }
    }
    return half_list;
}

// For test
vector3d NeighborHalf::get_diff_list(){
    vector3d diff_list(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        for (int k = offset[i]; k < offset[i+1]; ++k){
            diff_list[i].emplace_back(vector1d{dx[k], dy[k], dz[k]});
        }
    }
    return diff_list;
}

void NeighborHalf::get_full_list(
    vector1i& neigh_full,
    vector1d& dx_full,
    vector1d& dy_full,
    vector1d& dz_full,
    vector1i& offset_full){

    vector1i degree(n_total_atom, 0);
    for (int i = 0; i < n_total_atom; ++i) {
        degree[i] += size(i);
        auto [begin, end] = range(i);
        for (int k = begin; k < end; ++k) {
            int j = neighbor_atom(k);
            ++degree[j];
        }
    }

    offset_full = vector1i(n_total_atom + 1, 0);
    for (int i = 0; i < n_total_atom; ++i) {
        offset_full[i + 1] = offset_full[i] + degree[i];
    }

    int nnz = offset_full[n_total_atom];
    neigh_full = vector1i(nnz);
    dx_full = vector1d(nnz);
    dy_full = vector1d(nnz);
    dz_full = vector1d(nnz);

    vector1i pos(offset_full);
    for (int i = 0; i < n_total_atom; ++i) {
        auto [begin, end] = range(i);
        for (int k = begin; k < end; ++k) {
            int j = neighbor_atom(k);
            double dx, dy, dz;
            diff(k, dx, dy, dz);
            {
                int idx = pos[i];
                neigh_full[idx] = j;
                dx_full[idx] = dx;
                dy_full[idx] = dy;
                dz_full[idx] = dz;
                ++pos[i];
            }
            {
                int idx = pos[j];
                neigh_full[idx] = i;
                dx_full[idx] = -dx;
                dy_full[idx] = -dy;
                dz_full[idx] = -dz;
                ++pos[j];
            }
        }
    }
}
