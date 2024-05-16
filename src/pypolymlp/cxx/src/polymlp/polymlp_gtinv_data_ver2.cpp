/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_gtinv_data_ver2.h"

GtinvDataVer2::GtinvDataVer2(){

    set_gtinv_info();
}

GtinvDataVer2::~GtinvDataVer2(){}

const vector2i& GtinvDataVer2::get_l_array() const{ return l_array_all; }
const vector3i& GtinvDataVer2::get_m_array() const{ return m_array_all; }
const vector2d& GtinvDataVer2::get_coeffs() const{ return coeffs_all; }

void GtinvDataVer2::set_gtinv_info(){

}
