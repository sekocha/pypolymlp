/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_CELL
#define __NEIGHBOR_CELL

#include "mlpcpp.h"


class NeighborCell{

    vector2d trans_c_array;

    vector2d axis;
    double cutoff;

    vector1d axis0, axis1, axis2;
    double dot00, dot11, dot22, dot01, dot02, dot12;


    vector1d get_axis_vector(const int col);


    double dot_prod(const vector1d& vec1, const vector1d& vec2);
    //vector1d dot_prod(const vector2d& mat, const vector1i& vec);
    double norm(const vector1d& vec);
    vector1d to_cartesian(const int i, const int j, const int k);
    double distance(const int i, const int j, const int k);

    int refine_single_axis(
            const bool ref01, const bool ref02, const bool ref12,
            vector1i& vec0, vector1i& vec1, vector1i& vec2);

    int refine_axis(vector1i& vec0, vector1i& vec1, vector1i& vec2);

    double optimize_1d_axis0(const int val1, const int val2, vector1i& vec);
    double optimize_1d_axis1(const int val1, const int val2, vector1i& vec);
    double optimize_1d_axis2(const int val1, const int val2, vector1i& vec);

    double find_maximum_diagonal_in_cell();
    int find_trans();

    public: 

    NeighborCell(const vector2d& axis_i, const double cutoff_i);
    ~NeighborCell();

    const vector2d& get_translations() const;

};

#endif
