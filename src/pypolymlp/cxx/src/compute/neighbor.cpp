/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor.h"

Neighbor::Neighbor(const vector2d& axis, 
                   const vector2d& positions_c, 
                   const vector1i& types, 
                   const int& n_type, 
                   const double& cutoff){

    NeighborCell neigh_cell(axis, cutoff);
    const auto& trans = neigh_cell.get_translations();

    const int n_total_atom = types.size();
    dis_array = vector3d(n_total_atom, vector2d(n_type));
    diff_array = vector4d(n_total_atom, vector3d(n_type));
    atom2_array = vector3i(n_total_atom, vector2i(n_type));

    double dx, dy, dz;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < n_total_atom; ++j){
            for (const auto& tr: trans){
                dx = positions_c[0][j] + tr[0] - positions_c[0][i];
                dy = positions_c[1][j] + tr[1] - positions_c[1][i];
                dz = positions_c[2][j] + tr[2] - positions_c[2][i];
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    int type2 = types[j];
                    dis_array[i][type2].emplace_back(dis);
                    diff_array[i][type2].emplace_back(vector1d{dx,dy,dz});
                    atom2_array[i][type2].emplace_back(j);
                }
            }
        }
    }
}

Neighbor::~Neighbor(){}


const vector3d& Neighbor::get_dis_array() const{ return dis_array; }
const vector4d& Neighbor::get_diff_array() const{ return diff_array; }
const vector3i& Neighbor::get_atom2_array() const{ return atom2_array; }

