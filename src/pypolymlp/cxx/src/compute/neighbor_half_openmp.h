/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_HALF_OPENMP
#define __NEIGHBOR_HALF_OPENMP

#include "mlpcpp.h"
#include "neighbor_cell.h"

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

#endif
