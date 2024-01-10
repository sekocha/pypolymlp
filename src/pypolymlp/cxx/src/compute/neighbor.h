/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR
#define __NEIGHBOR

#include "mlpcpp.h"

class Neighbor{

    vector3d dis_array; 
    vector4d diff_array; 
    vector3i atom2_array;

    vector2d find_trans(const vector2d& axis, const double& cutoff);
    vector1d prod(const vector2d& mat, const vector1i& vec);

    public: 

    Neighbor(const vector2d& axis, 
             const vector2d& positions_c, 
             const vector1i& types, 
             const int& n_type, 
             const double& cutoff);

    ~Neighbor();

    const vector3d& get_dis_array() const;
    const vector4d& get_diff_array() const;
    const vector3i& get_atom2_array() const;

};

#endif
