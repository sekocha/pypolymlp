/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_GTINV_DATA
#define __POLYMLP_GTINV_DATA

#include "polymlp_mlpcpp.h"


#pragma once

template<int Order>
class GtinvData {

    static const vector2i L_ARRAY_ALL;
    static const vector3i M_ARRAY_ALL;
    static const vector2d COEFFS_ALL;

    public:

    GtinvData() = default;
    ~GtinvData() = default;

    static const vector2i& get_static_l_array() { return L_ARRAY_ALL; }
    static const vector3i& get_static_m_array() { return M_ARRAY_ALL; }
    static const vector2d& get_static_coeffs() { return COEFFS_ALL; }
};


#endif
