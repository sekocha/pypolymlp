/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_GTINV_DATA
#define __POLYMLP_GTINV_DATA

#include "polymlp_mlpcpp.h"
#include "polymlp_gtinv_data_order1.h"
#include "polymlp_gtinv_data_order2.h"
#include "polymlp_gtinv_data_order3.h"
#include "polymlp_gtinv_data_order4.h"
#include "polymlp_gtinv_data_order5.h"
#include "polymlp_gtinv_data_order6.h"

class GtinvData{

    vector2i l_array_all;
    vector3i m_array_all;
    vector2d coeffs_all;

    void set_gtinv_info();

    public:

    GtinvData();
   ~GtinvData();

    const vector2i& get_l_array() const;
    const vector3i& get_m_array() const;
    const vector2d& get_coeffs() const;

};

#endif
