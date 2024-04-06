/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_CELL
#define __NEIGHBOR_CELL

#include "mlpcpp.h"


class NeighborCell{

    vector2d trans_c_array;

    vector1d prod(const vector2d& mat, const vector1i& vec);
    double norm(const vector1d& vec);
    double distance(const vector2d& axis, const int i, const int j, const int k);

    double find_maximum_diagonal_in_cell(const vector2d& axis);
    int find_trans(const vector2d& axis, const double cutoff);

    public: 

    NeighborCell(const vector2d& axis, const double cutoff);
    ~NeighborCell();

    const vector2d& get_translations() const;

};

#endif
