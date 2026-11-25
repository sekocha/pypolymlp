/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_HALF_SINGLE
#define __NEIGHBOR_HALF_SINGLE

//#include <omp.h>
//#include <thread>

#include "mlpcpp.h"
#include "neighbor_cell.h"


class NeighborHalfSingle {

    vector1i offset;
    vector1i neigh;
    vector1d dx, dy, dz;

    int n_total_atom;

    public:

    NeighborHalfSingle(
        const vector2d& axis,
        const vector2d& positions_c,
        const double cutoff,
        const bool use_openmp
    );
    ~NeighborHalfSingle();

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

    vector2i get_half_list();
    vector3d get_diff_list();

    void get_full_list(
        vector1i& neigh_full,
        vector1d& dx_full,
        vector1d& dy_full,
        vector1d& dz_full,
        vector1i& offset_full);

};

#endif
