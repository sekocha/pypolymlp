/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_CELL
#define __NEIGHBOR_CELL

#include "mlpcpp.h"


class NeighborCell{

    vector2d trans_c_array;

    double dot_prod(const vector1d& vec1, const vector1d& vec2);
    vector1d dot_prod(const vector2d& mat, const vector1i& vec);
    double norm(const vector1d& vec);

    vector1d get_axis_vector(const vector2d& axis, const int col);
    double distance(const vector2d& axis, 
                    const int i, const int j, const int k);

    int refine_single_axis(
            const vector2d& axis, 
            const bool ref01, const bool ref02, const bool ref12,
            const double dot00, const double dot11, const double dot22,
            const double dot01, const double dot02, const double dot12,
            vector1i& vec0, vector1i& vec1, vector1i& vec2);

    int refine_axis(const vector2d& axis, 
                    const double dot00, 
                    const double dot11, 
                    const double dot22, 
                    const double dot01, 
                    const double dot02, 
                    const double dot12, 
                    vector1i& vec0, vector1i& vec1, vector1i& vec2);

    double optimize_1d_axis0(
        const vector2d& axis, const int val1, const int val2,
        vector1i& vec);
    double optimize_1d_axis1(
        const vector2d& axis, const int val1, const int val2,
        vector1i& vec);
    double optimize_1d_axis2(
        const vector2d& axis, const int val1, const int val2,
        vector1i& vec);

    double find_maximum_diagonal_in_cell(const vector2d& axis);
    int find_trans(const vector2d& axis, const double cutoff);

    public: 

    NeighborCell(const vector2d& axis, const double cutoff);
    ~NeighborCell();

    const vector2d& get_translations() const;

};

#endif
