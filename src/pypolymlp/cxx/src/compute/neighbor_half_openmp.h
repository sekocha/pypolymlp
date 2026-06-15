/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_HALF_OPENMP
#define __NEIGHBOR_HALF_OPENMP

#include "mlpcpp.h"
#include "neighbor_cell.h"


class NeighborHalfOpenMP {
private:
    vector1i offset;
    vector1i neigh;
    vector1d dx, dy, dz;

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

    inline int j(int i, int k) const {
        return neigh[offset[i] + k];
    }

    inline void diff(int i, int k, double &x, double &y, double &z) const {
        int id = offset[i] + k;
        x = dx[id];
        y = dy[id];
        z = dz[id];
    }
};
/*
class NeighborHalfOpenMP {
private:
    vector1i offset;
    vector1i neigh;

    vector1d dx, dy, dz;

public:
    NeighborHalfOpenMP(
        const vector2d& axis,
        const vector2d& positions_c,
        const vector1i& types,
        const double& cutoff
    );
    ~NeighborHalfOpenMP(){}

    inline int size(int i) const {
        return offset[i+1] - offset[i];
    }

    inline int get_neighbor(int i, int k) const {
        return neigh[offset[i] + k];
    }

    inline vector1d get_diff(int i, int k) const {
        int id = offset[i] + k;
        return {dx[id], dy[id], dz[id]};
    }

    const vector1i& get_offset() const { return offset; }
};
*/
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
