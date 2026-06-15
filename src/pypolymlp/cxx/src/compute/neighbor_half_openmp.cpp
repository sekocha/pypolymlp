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

    const double tol = 1e-12;
    const size_t n_total_atom = types.size();

    vector1i count(n_total_atom, 0);
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < i; ++j){
            double dx_ij = positions_c_rev[0][i] - positions_c_rev[0][j];
            double dy_ij = positions_c_rev[1][i] - positions_c_rev[1][j];
            double dz_ij = positions_c_rev[2][i] - positions_c_rev[2][j];
            for (const auto& tr: trans){
                double dx = dx_ij + tr[0];
                double dy = dy_ij + tr[1];
                double dz = dz_ij + tr[2];
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff && dis > tol){
                    count[i]++;
                }
            }
        }
        for (const auto& tr: trans){
            double dx = tr[0];
            double dy = tr[1];
            double dz = tr[2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < cutoff*cutoff && r2 > tol*tol){
                bool keep;
                if (dz >= tol) keep = true;
                else if (fabs(dz) < tol && dy >= tol) keep = true;
                else if (fabs(dz) < tol && fabs(dy) < tol && dx >= tol)
                    keep = true;
                else keep = false;
                if (keep){
                    count[i]++;
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

    vector1i pos = offset;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < i; ++j){
            double dx_ij = positions_c_rev[0][i] - positions_c_rev[0][j];
            double dy_ij = positions_c_rev[1][i] - positions_c_rev[1][j];
            double dz_ij = positions_c_rev[2][i] - positions_c_rev[2][j];
            for (const auto& tr: trans){
                double dxv = dx_ij + tr[0];
                double dyv = dy_ij + tr[1];
                double dzv = dz_ij + tr[2];
                double r2 = dxv*dxv + dyv*dyv + dzv*dzv;
                if (r2 < cutoff*cutoff && r2 > 1e-20){
                    int id = pos[i]++;
                    neigh[id] = j;
                    dx[id] = dxv;
                    dy[id] = dyv;
                    dz[id] = dzv;
                }
            }
        }
        // -------- self (i = j) --------
        for (const auto& tr: trans){
            double dxv = tr[0];
            double dyv = tr[1];
            double dzv = tr[2];
            double r2 = dxv*dxv + dyv*dyv + dzv*dzv;
            if (r2 < cutoff*cutoff && r2 > 1e-20){
                bool keep;
                if (dzv >= 1e-12) keep = true;
                else if (fabs(dzv) < 1e-12 && dyv >= 1e-12) keep = true;
                else if (fabs(dzv) < 1e-12 && fabs(dyv) < 1e-12 && dxv >= 1e-12)
                    keep = true;
                else keep = false;
                if (keep){
                    int id = pos[i]++;
                    neigh[id] = i;
                    dx[id] = dxv;
                    dy[id] = dyv;
                    dz[id] = dzv;
                }
            }
        }
    }

    /*
    half_list = vector2i(n_total_atom);
    diff_list = vector3d(n_total_atom);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int i = 0; i < n_total_atom; ++i){
        double dx, dy, dz, dx_ij, dy_ij, dz_ij, dis;
        bool bool_half;
        for (int j = 0; j < i; ++j){
            dx_ij = positions_c_rev[0][j] - positions_c_rev[0][i];
            dy_ij = positions_c_rev[1][j] - positions_c_rev[1][i];
            dz_ij = positions_c_rev[2][j] - positions_c_rev[2][i];
            for (const auto& tr: trans){
                dx = dx_ij + tr[0], dy = dy_ij + tr[1], dz = dz_ij + tr[2];
                dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    half_list[i].emplace_back(j);
                    diff_list[i].emplace_back(vector1d({dx, dy, dz}));
                }
            }
        }
        // j = i
        for (const auto& tr: trans){
            dx = tr[0], dy = tr[1], dz = tr[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < cutoff and dis > 1e-10){
                if (dz >= tol) bool_half = true;
                else if (fabs(dz) < tol and dy >= tol) bool_half = true;
                else if (fabs(dz) < tol and fabs(dy) < tol and dx >=tol)
                    bool_half = true;
                else bool_half = false;
                if (bool_half == true){
                    half_list[i].emplace_back(i);
                    diff_list[i].emplace_back(tr);
                }
            }
        }
    }
    */
}

NeighborHalfOpenMP::~NeighborHalfOpenMP(){}
//
//const vector2i& NeighborHalfOpenMP::get_half_list() const { return half_list; }
//const vector3d& NeighborHalfOpenMP::get_diff_list() const { return diff_list; }
