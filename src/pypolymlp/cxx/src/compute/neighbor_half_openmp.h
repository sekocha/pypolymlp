/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_HALF_OPENMP
#define __NEIGHBOR_HALF_OPENMP

#include "mlpcpp.h"
#include "neighbor_cell.h"


class NeighborHalfOpenMP {

    vector1i offset;
    vector1i neigh;
    vector1d dx, dy, dz;

    int n_total_atom;

    public:

    NeighborHalfOpenMP(
        const vector2d& axis,
        const vector2d& positions_c,
        const vector1i& types,
        const double& cutoff);
    ~NeighborHalfOpenMP();

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
        x = dx[id];
        y = dy[id];
        z = dz[id];
    }

    vector2i get_half_list();
    vector3d get_diff_list();
};
/*
class NeighborHalfOpenMP{

    vector2i half_list;
    vector3d diff_list;

    public:

    NeighborHalfOpenMP(
        const vector2d& axis,
        const vector2d& positions_c,
        const vector1i& types,
        const double& cutoff
    );

    ~NeighborHalfOpenMP();

    const vector2i& get_half_list() const;
    const vector3d& get_diff_list() const;

};
*/

#endif
