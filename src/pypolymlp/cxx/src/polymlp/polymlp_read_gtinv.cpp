/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_read_gtinv.h"


Readgtinv::Readgtinv(){}
Readgtinv::Readgtinv(
    const int gtinv_order,
    const vector1i& gtinv_maxl,
    const int version){

    screening(gtinv_order, gtinv_maxl, version);
}

Readgtinv::~Readgtinv(){}


void Readgtinv::screening(
    const int gtinv_order,
    const vector1i& gtinv_maxl,
    const int version){

    auto data_v2 = GtinvData();
    for (int order = 1; order <= gtinv_order; ++order){
        data_v2.parse(order, version);
        const auto& l_array_all = data_v2.get_l_array();
        const auto& m_array_all = data_v2.get_m_array();
        const auto& coeffs_all  = data_v2.get_coeffs();

        int maxl(0);
        if (order > 1) maxl = gtinv_maxl[order-2];

        for (size_t i = 0; i < l_array_all.size(); ++i){
            const vector1i& lcomb = l_array_all[i];
            if (maxl < *(lcomb.end()-1))
                continue;

            int l, m;
            vector2i vec1(m_array_all[i].size(), vector1i(order));
            for (size_t j = 0; j < m_array_all[i].size(); ++j){
                const auto& mcomb = m_array_all[i][j];
                for (int k = 0; k < order; ++k){
                    l = lcomb[k];
                    m = mcomb[k];
                    vec1[j][k] = l*l+l+m;
                }
            }
            l_array.emplace_back(lcomb);
            lm_array.emplace_back(vec1);
            coeffs.emplace_back(coeffs_all[i]);
        }
    }
}


const vector3i& Readgtinv::get_lm_seq() const{ return lm_array; }
const vector2i& Readgtinv::get_l_comb() const{ return l_array; }
const vector2d& Readgtinv::get_lm_coeffs() const{ return coeffs; }
