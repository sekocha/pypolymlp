/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_gtinv_data_ver2.h"

GtinvDataVer2::GtinvDataVer2(){}
GtinvDataVer2::~GtinvDataVer2(){}

const vector2i& GtinvDataVer2::get_l_array(const int order) const {
    if (order == 1) return order1.get_l_array();
    else if (order == 2) return order2.get_l_array();
    else if (order == 3) return order3.get_l_array();
    else if (order == 4) return order4.get_l_array();
    else if (order == 5) return order5.get_l_array();
    else if (order == 6) return order6.get_l_array();
}

const vector3i& GtinvDataVer2::get_m_array(const int order) const {

    if (order == 1) return order1.get_m_array();
    else if (order == 2) return order2.get_m_array();
    else if (order == 3) return order3.get_m_array();
    else if (order == 4) return order4.get_m_array();
    else if (order == 5) return order5.get_m_array();
    else if (order == 6) return order6.get_m_array();
}

const vector2d& GtinvDataVer2::get_coeffs(const int order) const {
    if (order == 1) return order1.get_coeffs();
    else if (order == 2) return order2.get_coeffs();
    else if (order == 3) return order3.get_coeffs();
    else if (order == 4) return order4.get_coeffs();
    else if (order == 5) return order5.get_coeffs();
    else if (order == 6) return order6.get_coeffs();
}
