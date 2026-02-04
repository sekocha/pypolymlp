/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_gtinv_data_ver2_order1.h"

GtinvDataVer2Order1::GtinvDataVer2Order1(){

    set_gtinv_info();
}

GtinvDataVer2Order1::~GtinvDataVer2Order1(){}

const vector2i& GtinvDataVer2Order1::get_l_array() const{ return l_array_all; }
const vector3i& GtinvDataVer2Order1::get_m_array() const{ return m_array_all; }
const vector2d& GtinvDataVer2Order1::get_coeffs() const{ return coeffs_all; }

void GtinvDataVer2Order1::set_gtinv_info(){

    l_array_all = {
        {0}
    };

    coeffs_all = {
        {1.0}
    };

    m_array_all = {
        {
        {0}
        }
    };

}
