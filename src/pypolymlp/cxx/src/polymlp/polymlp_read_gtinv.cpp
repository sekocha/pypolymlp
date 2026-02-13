/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_read_gtinv.h"


const vector2i& get_l_array(int order) {
    switch (order) {
        case 1: return GtinvData<1>::get_static_l_array();
        case 2: return GtinvData<2>::get_static_l_array();
        case 3: return GtinvData<3>::get_static_l_array();
        case 4: return GtinvData<4>::get_static_l_array();
        case 5: return GtinvData<5>::get_static_l_array();
        case 6: return GtinvData<6>::get_static_l_array();
        default: throw std::invalid_argument("Invalid order");
    }
}

const vector3i& get_m_array(int order) {
    switch (order) {
        case 1: return GtinvData<1>::get_static_m_array();
        case 2: return GtinvData<2>::get_static_m_array();
        case 3: return GtinvData<3>::get_static_m_array();
        case 4: return GtinvData<4>::get_static_m_array();
        case 5: return GtinvData<5>::get_static_m_array();
        case 6: return GtinvData<6>::get_static_m_array();
        default: throw std::invalid_argument("Invalid order");
    }
}

const vector2d& get_coeffs(int order) {
    switch (order) {
        case 1: return GtinvData<1>::get_static_coeffs();
        case 2: return GtinvData<2>::get_static_coeffs();
        case 3: return GtinvData<3>::get_static_coeffs();
        case 4: return GtinvData<4>::get_static_coeffs();
        case 5: return GtinvData<5>::get_static_coeffs();
        case 6: return GtinvData<6>::get_static_coeffs();
        default: throw std::invalid_argument("Invalid order");
    }
}


Readgtinv::Readgtinv(){}
Readgtinv::Readgtinv(
    const int& gtinv_order,
    const vector1i& gtinv_maxl,
    const std::vector<bool>& gtinv_sym,
    const int& n_type,
    const int& version){

    if (version == 1)
        screening(gtinv_order, gtinv_maxl, gtinv_sym, n_type);
//    else if (version == 2)
//        screening_ver2(gtinv_order, gtinv_maxl, n_type);

}

Readgtinv::~Readgtinv(){}


void Readgtinv::screening(
    const int& gtinv_order,
    const vector1i& gtinv_maxl,
    const std::vector<bool>& gtinv_sym,
    const int& n_type){

    for (int order = 1; order <= gtinv_order; ++order){
        const auto& l_array_all = get_l_array(order);
        const auto& m_array_all = get_m_array(order);
        const auto& coeffs_all = get_coeffs(order);
        for (size_t i = 0; i < l_array_all.size(); ++i){
            const vector1i &lcomb = l_array_all[i];
            const int maxl = *(lcomb.end()-1);
            bool include(true);
            if (order > 1){
                if (maxl > gtinv_maxl[order-2]) include = false;
                if (gtinv_sym[order-2] == true){
                    int n_ele = std::count(lcomb.begin(), lcomb.end(), lcomb[0]);
                    if (n_ele != order) include = false;
                }
            }

            if (include == true){
                int l, m;
                vector2i vec1(m_array_all[i].size(), vector1i(order));
                for (size_t j = 0; j < m_array_all[i].size(); ++j){
                    const auto &mcomb = m_array_all[i][j];
                    for (int k = 0; k < order; ++k){
                        l = lcomb[k], m = mcomb[k];
                        vec1[j][k] = l*l+l+m;
                    }
                }
                l_array.emplace_back(lcomb);
                lm_array.emplace_back(vec1);
                coeffs.emplace_back(coeffs_all[i]);
            }
        }
    }
}

const vector3i& Readgtinv::get_lm_seq() const{ return lm_array; }
const vector2i& Readgtinv::get_l_comb() const{ return l_array; }
const vector2d& Readgtinv::get_lm_coeffs() const{ return coeffs; }

/*
void Readgtinv::screening_ver2(const int& gtinv_order,
                               const vector1i& gtinv_maxl,
                               const int& n_type){

    GtinvDataVer2 data;
    for (int order = 1; order < gtinv_order + 1; ++order){
        const auto& l_array_all = data.get_l_array(order);
        const auto& m_array_all = data.get_m_array(order);
        const auto& coeffs_all = data.get_coeffs(order);

        for (size_t i = 0; i < l_array_all.size(); ++i){
            const vector1i& lcomb = l_array_all[i];
            bool tag = true;
            const int maxl = *(lcomb.end()-1);
            if (order > 1){
                if (maxl > gtinv_maxl[order-2]) tag = false;
            }

            if (tag == true){
                int l, m;
                vector2i vec1(m_array_all[i].size(), vector1i(order));
                for (size_t j = 0; j < m_array_all[i].size(); ++j){
                    const auto &mcomb = m_array_all[i][j];
                    for (int k = 0; k < order; ++k){
                        l = lcomb[k], m = mcomb[k];
                        vec1[j][k] = l*l+l+m;
                    }
                }
                l_array.emplace_back(lcomb);
                lm_array.emplace_back(vec1);
                coeffs.emplace_back(coeffs_all[i]);
            }
        }
    }
}
*/
