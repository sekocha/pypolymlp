/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor.h"

Neighbor::Neighbor(
    const vector2d& axis,
    const vector2d& positions_c,
    const vector1i& types,
    const int& n_type,
    const double& cutoff
){

    NeighborCell neigh_cell(axis, positions_c, cutoff);
    const auto& trans = neigh_cell.get_translations();

    const int n_total_atom = types.size();
    dis_array = vector3d(n_total_atom, vector2d(n_type));
    diff_array = vector4d(n_total_atom, vector3d(n_type));
    atom2_array = vector3i(n_total_atom, vector2i(n_type));

    double dx, dy, dz, dx_ij, dy_ij, dz_ij, dis;
    for (int i = 0; i < n_total_atom; ++i){
        int type1 = types[i];
        for (int j = 0; j <= i; ++j){
            dx_ij = positions_c[0][j] - positions_c[0][i];
            dy_ij = positions_c[1][j] - positions_c[1][i];
            dz_ij = positions_c[2][j] - positions_c[2][i];
            int type2 = types[j];
            auto& dis_ij = dis_array[i][type2];
            auto& diff_ij = diff_array[i][type2];
            auto& atom2_ij = atom2_array[i][type2];
            auto& dis_ij_rev = dis_array[j][type1];
            auto& diff_ij_rev = diff_array[j][type1];
            auto& atom2_ij_rev = atom2_array[j][type1];
            for (const auto& tr: trans){
                dx = dx_ij + tr[0], dy = dy_ij + tr[1], dz = dz_ij + tr[2];
                dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    dis_ij.emplace_back(dis);
                    diff_ij.emplace_back(vector1d{dx,dy,dz});
                    atom2_ij.emplace_back(j);
                    if (i != j){
                        dis_ij_rev.emplace_back(dis);
                        diff_ij_rev.emplace_back(vector1d{-dx,-dy,-dz});
                        atom2_ij_rev.emplace_back(i);
                    }
                }
            }
        }
    }
}

Neighbor::~Neighbor(){}


const vector3d& Neighbor::get_dis_array() const{ return dis_array; }
const vector4d& Neighbor::get_diff_array() const{ return diff_array; }
const vector3i& Neighbor::get_atom2_array() const{ return atom2_array; }
