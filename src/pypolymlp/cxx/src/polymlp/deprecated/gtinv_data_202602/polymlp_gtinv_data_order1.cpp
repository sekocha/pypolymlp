/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_gtinv_data_order1.h"

GtinvDataOrder1::GtinvDataOrder1(){

    set_gtinv_info();
}

GtinvDataOrder1::~GtinvDataOrder1(){}

const vector2i& GtinvDataOrder1::get_l_array() const{ return l_array_all; }
const vector3i& GtinvDataOrder1::get_m_array() const{ return m_array_all; }
const vector2d& GtinvDataOrder1::get_coeffs() const{ return coeffs_all; }

void GtinvDataOrder1::set_gtinv_info(){

    l_array_all =
        {{0}};

    coeffs_all =
        {{1}};

    m_array_all = {
        {{0}}
    };
}
