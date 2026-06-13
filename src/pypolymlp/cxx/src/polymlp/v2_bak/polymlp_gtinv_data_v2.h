/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_GTINV_DATA_VER2
#define __POLYMLP_GTINV_DATA_VER2

#include "polymlp_mlpcpp.h"


#pragma once

template<int Order, int L>
class GtinvDataVer2 {

    static const vector2i L_ARRAY_ALL;
    static const vector3i M_ARRAY_ALL;
    static const vector2d COEFFS_ALL;

    public:

    GtinvDataVer2() = default;
    ~GtinvDataVer2() = default;

    static const vector2i& get_static_l_array() { return L_ARRAY_ALL; }
    static const vector3i& get_static_m_array() { return M_ARRAY_ALL; }
    static const vector2d& get_static_coeffs() { return COEFFS_ALL; }
};


#endif
