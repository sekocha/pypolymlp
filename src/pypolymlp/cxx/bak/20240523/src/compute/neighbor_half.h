/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_HALF
#define __NEIGHBOR_HALF

#include "mlpcpp.h"
#include "neighbor_cell.h"

class NeighborHalf{

    vector2i half_list;
    vector3d diff_list;

    public:

    NeighborHalf(const vector2d& axis,
                 const vector2d& positions_c,
                 const vector1i& types,
                 const double& cutoff);

    ~NeighborHalf();

    const vector2i& get_half_list() const;
    const vector3d& get_diff_list() const;

};

#endif
