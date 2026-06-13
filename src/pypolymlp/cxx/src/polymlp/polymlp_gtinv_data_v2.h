/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_GTINV_DATA_VER2
#define __POLYMLP_GTINV_DATA_VER2

#include "polymlp_gtinv_binary.h"
#include "polymlp_mlpcpp.h"


#include <filesystem>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif


class GtinvDataVer2 {

    vector2i32 l_array_all;
    vector2d coeffs_all;
    vector3i32 m_array_all;

    public:

    GtinvDataVer2();
    ~GtinvDataVer2();
    void parse(const int order);

    const vector2i32& get_l_array() const;
    const vector3i32& get_m_array() const;
    const vector2d& get_coeffs() const;
};

#endif
