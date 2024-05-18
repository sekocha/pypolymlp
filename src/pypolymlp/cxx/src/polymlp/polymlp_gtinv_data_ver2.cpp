/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_gtinv_data_ver2.h"

GtinvDataVer2::GtinvDataVer2(){

    GtinvDataVer2Order1 order1;
    for (const auto& a: order1.get_l_array()) l_array_all.emplace_back(a);
    for (const auto& a: order1.get_m_array()) m_array_all.emplace_back(a);
    for (const auto& a: order1.get_coeffs()) coeffs_all.emplace_back(a);

    GtinvDataVer2Order2 order2;
    for (const auto& a: order2.get_l_array()) l_array_all.emplace_back(a);
    for (const auto& a: order2.get_m_array()) m_array_all.emplace_back(a);
    for (const auto& a: order2.get_coeffs()) coeffs_all.emplace_back(a);

    GtinvDataVer2Order3 order3;
    for (const auto& a: order3.get_l_array()) l_array_all.emplace_back(a);
    for (const auto& a: order3.get_m_array()) m_array_all.emplace_back(a);
    for (const auto& a: order3.get_coeffs()) coeffs_all.emplace_back(a);

    GtinvDataVer2Order4 order4;
    for (const auto& a: order4.get_l_array()) l_array_all.emplace_back(a);
    for (const auto& a: order4.get_m_array()) m_array_all.emplace_back(a);
    for (const auto& a: order4.get_coeffs()) coeffs_all.emplace_back(a);

    GtinvDataVer2Order5 order5;
    for (const auto& a: order5.get_l_array()) l_array_all.emplace_back(a);
    for (const auto& a: order5.get_m_array()) m_array_all.emplace_back(a);
    for (const auto& a: order5.get_coeffs()) coeffs_all.emplace_back(a);

    GtinvDataVer2Order6 order6;
    for (const auto& a: order6.get_l_array()) l_array_all.emplace_back(a);
    for (const auto& a: order6.get_m_array()) m_array_all.emplace_back(a);
    for (const auto& a: order6.get_coeffs()) coeffs_all.emplace_back(a);

}

GtinvDataVer2::~GtinvDataVer2(){}

const vector2i& GtinvDataVer2::get_l_array() const{ return l_array_all; }
const vector3i& GtinvDataVer2::get_m_array() const{ return m_array_all; }
const vector2d& GtinvDataVer2::get_coeffs() const{ return coeffs_all; }


