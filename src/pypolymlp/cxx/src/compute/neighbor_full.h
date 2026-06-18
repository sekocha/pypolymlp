/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_FULL
#define __NEIGHBOR_FULL

#include <omp.h>
#include <thread>

#include "mlpcpp.h"
#include "neighbor_cell.h"


class NeighborFull {

    vector1i offset;
    vector1i neigh;
    vector1d dx, dy, dz;

    int n_total_atom;

    public:

    NeighborFull(
        const vector2d& axis,
        const vector2d& positions_c,
        const double cutoff
    );
    ~NeighborFull();

    inline int size(int i) const {
        return offset[i+1] - offset[i];
    }

    inline std::pair<int,int> range(int i) const {
        return {offset[i], offset[i+1]};
    }

    inline int neighbor_atom(int id) const {
        return neigh[id];
    }

    inline void diff(int id, double &x, double &y, double &z) const {
        // (dx, dy, dz) = pos[j] - pos[i]
        x = dx[id];
        y = dy[id];
        z = dz[id];
    }

    inline void diff_ji(int id, double &x, double &y, double &z) const {
        // (dx, dy, dz) = pos[j] - pos[i]
        x = dx[id];
        y = dy[id];
        z = dz[id];
    }

    inline void diff_ij(int id, double &x, double &y, double &z) const {
        // (dx, dy, dz) = pos[i] - pos[j]
        x = - dx[id];
        y = - dy[id];
        z = - dz[id];
    }

    // For test
    vector3d get_dis_array(const int n_type, const vector1i& types);
    vector4d get_diff_array(const int n_type, const vector1i& types);
    vector3i get_atom2_array(const int n_type, const vector1i& types);

};

#endif
