/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_half_openmp.h"


NeighborHalfOpenMP::NeighborHalfOpenMP(
    const vector2d& axis,
    const vector2d& positions_c,
    const vector1i& types,
    const double& cutoff
){

    NeighborCell neigh_cell(axis, positions_c, cutoff);
    const auto& trans = neigh_cell.get_translations();
    const auto& positions_c_rev = neigh_cell.get_positions_cartesian();

    n_total_atom = positions_c[0].size();
    const double tol = 1e-10;
    const double tol_sq = tol * tol;
    const double cutoff_sq = cutoff * cutoff;

    vector1i count(n_total_atom, 0);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < i; ++j){
            double dx_ij = positions_c_rev[0][j] - positions_c_rev[0][i];
            double dy_ij = positions_c_rev[1][j] - positions_c_rev[1][i];
            double dz_ij = positions_c_rev[2][j] - positions_c_rev[2][i];
            for (const auto& tr: trans){
                double dx = dx_ij + tr[0];
                double dy = dy_ij + tr[1];
                double dz = dz_ij + tr[2];
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < cutoff_sq and r2 > tol_sq) ++count[i];
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
                    ++count[i];
                }
            }
        }
    }

    offset.resize(n_total_atom + 1);
    offset[0] = 0;
    for (int i = 0; i < n_total_atom; ++i){
        offset[i+1] = offset[i] + count[i];
    }

    int N = offset[n_total_atom];
    neigh.resize(N);
    dx.resize(N);
    dy.resize(N);
    dz.resize(N);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int i = 0; i < n_total_atom; ++i){
        int id = offset[i];
        for (int j = 0; j < i; ++j){
            double dx_ij = positions_c_rev[0][j] - positions_c_rev[0][i];
            double dy_ij = positions_c_rev[1][j] - positions_c_rev[1][i];
            double dz_ij = positions_c_rev[2][j] - positions_c_rev[2][i];
            for (const auto& tr: trans){
                double dxv = dx_ij + tr[0];
                double dyv = dy_ij + tr[1];
                double dzv = dz_ij + tr[2];
                double r2 = dxv*dxv + dyv*dyv + dzv*dzv;
                if (r2 < cutoff_sq and r2 > tol_sq){
                    neigh[id] = j;
                    dx[id] = dxv;
                    dy[id] = dyv;
                    dz[id] = dzv;
                    ++id;
                }
            }
        }
        // -------- self (i = j) --------
        for (const auto& tr: trans){
            double dxv = tr[0];
            double dyv = tr[1];
            double dzv = tr[2];
            double r2 = dxv*dxv + dyv*dyv + dzv*dzv;
            if (r2 < cutoff_sq and r2 > tol_sq){
                bool keep;
                if (dzv >= tol) keep = true;
                else if (fabs(dzv) < tol and dyv >= tol) keep = true;
                else if (fabs(dzv) < tol and fabs(dyv) < tol and dxv >= tol)
                    keep = true;
                else keep = false;
                if (keep){
                    neigh[id] = i;
                    dx[id] = dxv;
                    dy[id] = dyv;
                    dz[id] = dzv;
                    ++id;
                }
            }
        }
    }
}

NeighborHalfOpenMP::~NeighborHalfOpenMP(){}

// For test
vector2i NeighborHalfOpenMP::get_half_list(){
    vector2i half_list(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        for (int k = offset[i]; k < offset[i+1]; ++k){
            half_list[i].emplace_back(neigh[k]);
        }
    }
    return half_list;
}

// For test
vector3d NeighborHalfOpenMP::get_diff_list(){
    vector3d diff_list(n_total_atom);
    for (int i = 0; i < n_total_atom; ++i){
        for (int k = offset[i]; k < offset[i+1]; ++k){
            diff_list[i].emplace_back(vector1d{dx[k], dy[k], dz[k]});
        }
    }
    return diff_list;
}
