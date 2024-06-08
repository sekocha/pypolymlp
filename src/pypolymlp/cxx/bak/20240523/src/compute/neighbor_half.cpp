/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_half.h"
//#include <chrono>


NeighborHalf::NeighborHalf(const vector2d& axis,
                           const vector2d& positions_c,
                           const vector1i& types,
                           const double& cutoff){

//    auto t1 = std::chrono::system_clock::now();

    NeighborCell neigh_cell(axis, positions_c, cutoff);
    const auto& trans = neigh_cell.get_translations();
    const auto& positions_c_rev = neigh_cell.get_positions_cartesian();
/*
    auto t2 = std::chrono::system_clock::now();
    auto dur = t2 - t1;
    auto msec = std::chrono::duration_cast
                <std::chrono::microseconds>(dur).count();
    std::cout << msec << std::endl;
    auto t3 = std::chrono::system_clock::now();
*/
    double dx, dy, dz, dx_ij, dy_ij, dz_ij, dis;
    bool bool_half;
    const double tol = 1e-12;
    const size_t n_total_atom = types.size();
    half_list = vector2i(n_total_atom);
    diff_list = vector3d(n_total_atom);

    for (int i = 0; i < n_total_atom; ++i){
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

/*
    vector1i count(n_total_atom, 0);
    for (int i = 0; i < n_total_atom; ++i){
        for (auto& j: half_list[i]){
            count[j] += 1;
            count[i] += 1;
        }
    }
    for (int i = 0; i < n_total_atom; ++i){
        std::cout << i << " " << count[i] << std::endl;
    }
*/

/*
    auto t4 = std::chrono::system_clock::now();
    dur = t4 - t3;
    msec = std::chrono::duration_cast
                <std::chrono::microseconds>(dur).count();
    std::cout << msec << std::endl;
*/
}

NeighborHalf::~NeighborHalf(){}

const vector2i& NeighborHalf::get_half_list() const { return half_list; }
const vector3d& NeighborHalf::get_diff_list() const { return diff_list; }



/* lammps convention for pair choice
NeighborHalf::NeighborHalf(const vector2d& axis,
                           const vector2d& positions_c,
                           const vector1i& types,
                           const double& cutoff){

    const double tol = 1e-12;
    const auto& trans = find_trans(axis, cutoff);

    const int n_total_atom = types.size();
    half_list = vector2i(n_total_atom);
    diff_list = vector3d(n_total_atom);

    double dx, dy, dz;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < n_total_atom; ++j){
            for (const auto& tr: trans){
                dx = positions_c[0][j] + tr[0] - positions_c[0][i];
                dy = positions_c[1][j] + tr[1] - positions_c[1][i];
                dz = positions_c[2][j] + tr[2] - positions_c[2][i];
                bool bool_half = false;
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    if (dz >= tol)
                        bool_half = true;
                    else if (fabs(dz) < tol and dy >= tol)
                        bool_half = true;
                    else if (fabs(dz) < tol and fabs(dy) < tol and dx >=tol)
                        bool_half = true;
                }
                if (bool_half == true){
                    half_list[i].emplace_back(j);
                    diff_list[i].emplace_back(vector1d({dx, dy, dz}));
                }
            }
        }
    }
}
*/
