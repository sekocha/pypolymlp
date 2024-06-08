/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_GTINV_DATA_VER2_ORDER5
#define __POLYMLP_GTINV_DATA_VER2_ORDER5

#include "polymlp_mlpcpp.h"

class GtinvDataVer2Order5{

    vector2i l_array_all;
    vector3i m_array_all;
    vector2d coeffs_all;

    void set_gtinv_info();

    public:

    GtinvDataVer2Order5();
   ~GtinvDataVer2Order5();

    const vector2i& get_l_array() const;
    const vector3i& get_m_array() const;
    const vector2d& get_coeffs() const;

};

#endif
