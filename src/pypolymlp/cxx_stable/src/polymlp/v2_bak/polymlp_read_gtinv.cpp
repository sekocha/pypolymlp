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

const vector2i& get_l_array_v2(int order, int l) {
    switch (order) {
        case 1:
            switch(l){
                case 0: return GtinvDataVer2<1,0>::get_static_l_array();
                default: throw std::invalid_argument("Invalid L");
            }
        case 2:
            switch (l) {
                case 0:  return GtinvDataVer2<2,0>::get_static_l_array();
                case 1:  return GtinvDataVer2<2,1>::get_static_l_array();
                case 2:  return GtinvDataVer2<2,2>::get_static_l_array();
                case 3:  return GtinvDataVer2<2,3>::get_static_l_array();
                case 4:  return GtinvDataVer2<2,4>::get_static_l_array();
                case 5:  return GtinvDataVer2<2,5>::get_static_l_array();
                case 6:  return GtinvDataVer2<2,6>::get_static_l_array();
                case 7:  return GtinvDataVer2<2,7>::get_static_l_array();
                case 8:  return GtinvDataVer2<2,8>::get_static_l_array();
                case 9:  return GtinvDataVer2<2,9>::get_static_l_array();
                case 10: return GtinvDataVer2<2,10>::get_static_l_array();
                case 11: return GtinvDataVer2<2,11>::get_static_l_array();
                case 12: return GtinvDataVer2<2,12>::get_static_l_array();
                case 13: return GtinvDataVer2<2,13>::get_static_l_array();
                case 14: return GtinvDataVer2<2,14>::get_static_l_array();
                case 15: return GtinvDataVer2<2,15>::get_static_l_array();
                case 16: return GtinvDataVer2<2,16>::get_static_l_array();
                case 17: return GtinvDataVer2<2,17>::get_static_l_array();
                case 18: return GtinvDataVer2<2,18>::get_static_l_array();
                case 19: return GtinvDataVer2<2,19>::get_static_l_array();
                case 20: return GtinvDataVer2<2,20>::get_static_l_array();
                case 21: return GtinvDataVer2<2,21>::get_static_l_array();
                case 22: return GtinvDataVer2<2,22>::get_static_l_array();
                case 23: return GtinvDataVer2<2,23>::get_static_l_array();
                case 24: return GtinvDataVer2<2,24>::get_static_l_array();
                case 25: return GtinvDataVer2<2,25>::get_static_l_array();
                case 26: return GtinvDataVer2<2,26>::get_static_l_array();
                case 27: return GtinvDataVer2<2,27>::get_static_l_array();
                case 28: return GtinvDataVer2<2,28>::get_static_l_array();
                case 29: return GtinvDataVer2<2,29>::get_static_l_array();
                case 30: return GtinvDataVer2<2,30>::get_static_l_array();
                default:
                    throw std::invalid_argument("Invalid L");
            }
        case 3:
            switch (l) {
                case 0:  return GtinvDataVer2<3,0>::get_static_l_array();
                case 1:  return GtinvDataVer2<3,1>::get_static_l_array();
                case 2:  return GtinvDataVer2<3,2>::get_static_l_array();
                case 3:  return GtinvDataVer2<3,3>::get_static_l_array();
                case 4:  return GtinvDataVer2<3,4>::get_static_l_array();
                case 5:  return GtinvDataVer2<3,5>::get_static_l_array();
                case 6:  return GtinvDataVer2<3,6>::get_static_l_array();
                case 7:  return GtinvDataVer2<3,7>::get_static_l_array();
                case 8:  return GtinvDataVer2<3,8>::get_static_l_array();
                case 9:  return GtinvDataVer2<3,9>::get_static_l_array();
                case 10: return GtinvDataVer2<3,10>::get_static_l_array();
                case 11: return GtinvDataVer2<3,11>::get_static_l_array();
                case 12: return GtinvDataVer2<3,12>::get_static_l_array();
                case 13: return GtinvDataVer2<3,13>::get_static_l_array();
                case 14: return GtinvDataVer2<3,14>::get_static_l_array();
                case 15: return GtinvDataVer2<3,15>::get_static_l_array();
                case 16: return GtinvDataVer2<3,16>::get_static_l_array();
                case 17: return GtinvDataVer2<3,17>::get_static_l_array();
                case 18: return GtinvDataVer2<3,18>::get_static_l_array();
                case 19: return GtinvDataVer2<3,19>::get_static_l_array();
                case 20: return GtinvDataVer2<3,20>::get_static_l_array();
                default:
                    throw std::invalid_argument("Invalid L. (L3 > 20)");
            }


        //case 3: return GtinvData<3>::get_static_coeffs();
        //case 4: return GtinvData<4>::get_static_coeffs();
        //case 5: return GtinvData<5>::get_static_coeffs();
        //case 6: return GtinvData<6>::get_static_coeffs();
        default: throw std::invalid_argument("Invalid order");
    }
}

const vector3i& get_m_array_v2(int order, int l) {
    switch (order) {
        case 1:
            switch(l){
                case 0: return GtinvDataVer2<1,0>::get_static_m_array();
                default: throw std::invalid_argument("Invalid L");
            }
        case 2:
            switch (l) {
                case 0:  return GtinvDataVer2<2,0>::get_static_m_array();
                case 1:  return GtinvDataVer2<2,1>::get_static_m_array();
                case 2:  return GtinvDataVer2<2,2>::get_static_m_array();
                case 3:  return GtinvDataVer2<2,3>::get_static_m_array();
                case 4:  return GtinvDataVer2<2,4>::get_static_m_array();
                case 5:  return GtinvDataVer2<2,5>::get_static_m_array();
                case 6:  return GtinvDataVer2<2,6>::get_static_m_array();
                case 7:  return GtinvDataVer2<2,7>::get_static_m_array();
                case 8:  return GtinvDataVer2<2,8>::get_static_m_array();
                case 9:  return GtinvDataVer2<2,9>::get_static_m_array();
                case 10: return GtinvDataVer2<2,10>::get_static_m_array();
                case 11: return GtinvDataVer2<2,11>::get_static_m_array();
                case 12: return GtinvDataVer2<2,12>::get_static_m_array();
                case 13: return GtinvDataVer2<2,13>::get_static_m_array();
                case 14: return GtinvDataVer2<2,14>::get_static_m_array();
                case 15: return GtinvDataVer2<2,15>::get_static_m_array();
                case 16: return GtinvDataVer2<2,16>::get_static_m_array();
                case 17: return GtinvDataVer2<2,17>::get_static_m_array();
                case 18: return GtinvDataVer2<2,18>::get_static_m_array();
                case 19: return GtinvDataVer2<2,19>::get_static_m_array();
                case 20: return GtinvDataVer2<2,20>::get_static_m_array();
                case 21: return GtinvDataVer2<2,21>::get_static_m_array();
                case 22: return GtinvDataVer2<2,22>::get_static_m_array();
                case 23: return GtinvDataVer2<2,23>::get_static_m_array();
                case 24: return GtinvDataVer2<2,24>::get_static_m_array();
                case 25: return GtinvDataVer2<2,25>::get_static_m_array();
                case 26: return GtinvDataVer2<2,26>::get_static_m_array();
                case 27: return GtinvDataVer2<2,27>::get_static_m_array();
                case 28: return GtinvDataVer2<2,28>::get_static_m_array();
                case 29: return GtinvDataVer2<2,29>::get_static_m_array();
                case 30: return GtinvDataVer2<2,30>::get_static_m_array();
                default:
                    throw std::invalid_argument("Invalid L");
            }
        case 3:
            switch (l) {
                case 0:  return GtinvDataVer2<3,0>::get_static_m_array();
                case 1:  return GtinvDataVer2<3,1>::get_static_m_array();
                case 2:  return GtinvDataVer2<3,2>::get_static_m_array();
                case 3:  return GtinvDataVer2<3,3>::get_static_m_array();
                case 4:  return GtinvDataVer2<3,4>::get_static_m_array();
                case 5:  return GtinvDataVer2<3,5>::get_static_m_array();
                case 6:  return GtinvDataVer2<3,6>::get_static_m_array();
                case 7:  return GtinvDataVer2<3,7>::get_static_m_array();
                case 8:  return GtinvDataVer2<3,8>::get_static_m_array();
                case 9:  return GtinvDataVer2<3,9>::get_static_m_array();
                case 10: return GtinvDataVer2<3,10>::get_static_m_array();
                case 11: return GtinvDataVer2<3,11>::get_static_m_array();
                case 12: return GtinvDataVer2<3,12>::get_static_m_array();
                case 13: return GtinvDataVer2<3,13>::get_static_m_array();
                case 14: return GtinvDataVer2<3,14>::get_static_m_array();
                case 15: return GtinvDataVer2<3,15>::get_static_m_array();
                case 16: return GtinvDataVer2<3,16>::get_static_m_array();
                case 17: return GtinvDataVer2<3,17>::get_static_m_array();
                case 18: return GtinvDataVer2<3,18>::get_static_m_array();
                case 19: return GtinvDataVer2<3,19>::get_static_m_array();
                case 20: return GtinvDataVer2<3,20>::get_static_m_array();
                default:
                    throw std::invalid_argument("Invalid L. (L3 > 20)");
            }


        //case 3: return GtinvData<3>::get_static_coeffs();
        //case 4: return GtinvData<4>::get_static_coeffs();
        //case 5: return GtinvData<5>::get_static_coeffs();
        //case 6: return GtinvData<6>::get_static_coeffs();
        default: throw std::invalid_argument("Invalid order");
    }
}

const vector2d& get_coeffs_v2(int order, int l) {
    switch (order) {
        case 1:
            switch(l){
                case 0: return GtinvDataVer2<1,0>::get_static_coeffs();
                default: throw std::invalid_argument("Invalid L");
            }
        case 2:
            switch (l) {
                case 0:  return GtinvDataVer2<2,0>::get_static_coeffs();
                case 1:  return GtinvDataVer2<2,1>::get_static_coeffs();
                case 2:  return GtinvDataVer2<2,2>::get_static_coeffs();
                case 3:  return GtinvDataVer2<2,3>::get_static_coeffs();
                case 4:  return GtinvDataVer2<2,4>::get_static_coeffs();
                case 5:  return GtinvDataVer2<2,5>::get_static_coeffs();
                case 6:  return GtinvDataVer2<2,6>::get_static_coeffs();
                case 7:  return GtinvDataVer2<2,7>::get_static_coeffs();
                case 8:  return GtinvDataVer2<2,8>::get_static_coeffs();
                case 9:  return GtinvDataVer2<2,9>::get_static_coeffs();
                case 10: return GtinvDataVer2<2,10>::get_static_coeffs();
                case 11: return GtinvDataVer2<2,11>::get_static_coeffs();
                case 12: return GtinvDataVer2<2,12>::get_static_coeffs();
                case 13: return GtinvDataVer2<2,13>::get_static_coeffs();
                case 14: return GtinvDataVer2<2,14>::get_static_coeffs();
                case 15: return GtinvDataVer2<2,15>::get_static_coeffs();
                case 16: return GtinvDataVer2<2,16>::get_static_coeffs();
                case 17: return GtinvDataVer2<2,17>::get_static_coeffs();
                case 18: return GtinvDataVer2<2,18>::get_static_coeffs();
                case 19: return GtinvDataVer2<2,19>::get_static_coeffs();
                case 20: return GtinvDataVer2<2,20>::get_static_coeffs();
                case 21: return GtinvDataVer2<2,21>::get_static_coeffs();
                case 22: return GtinvDataVer2<2,22>::get_static_coeffs();
                case 23: return GtinvDataVer2<2,23>::get_static_coeffs();
                case 24: return GtinvDataVer2<2,24>::get_static_coeffs();
                case 25: return GtinvDataVer2<2,25>::get_static_coeffs();
                case 26: return GtinvDataVer2<2,26>::get_static_coeffs();
                case 27: return GtinvDataVer2<2,27>::get_static_coeffs();
                case 28: return GtinvDataVer2<2,28>::get_static_coeffs();
                case 29: return GtinvDataVer2<2,29>::get_static_coeffs();
                case 30: return GtinvDataVer2<2,30>::get_static_coeffs();
                default:
                    throw std::invalid_argument("Invalid L. (L2 > 30)");
            }
        case 3:
            switch (l) {
                case 0:  return GtinvDataVer2<3,0>::get_static_coeffs();
                case 1:  return GtinvDataVer2<3,1>::get_static_coeffs();
                case 2:  return GtinvDataVer2<3,2>::get_static_coeffs();
                case 3:  return GtinvDataVer2<3,3>::get_static_coeffs();
                case 4:  return GtinvDataVer2<3,4>::get_static_coeffs();
                case 5:  return GtinvDataVer2<3,5>::get_static_coeffs();
                case 6:  return GtinvDataVer2<3,6>::get_static_coeffs();
                case 7:  return GtinvDataVer2<3,7>::get_static_coeffs();
                case 8:  return GtinvDataVer2<3,8>::get_static_coeffs();
                case 9:  return GtinvDataVer2<3,9>::get_static_coeffs();
                case 10: return GtinvDataVer2<3,10>::get_static_coeffs();
                case 11: return GtinvDataVer2<3,11>::get_static_coeffs();
                case 12: return GtinvDataVer2<3,12>::get_static_coeffs();
                case 13: return GtinvDataVer2<3,13>::get_static_coeffs();
                case 14: return GtinvDataVer2<3,14>::get_static_coeffs();
                case 15: return GtinvDataVer2<3,15>::get_static_coeffs();
                case 16: return GtinvDataVer2<3,16>::get_static_coeffs();
                case 17: return GtinvDataVer2<3,17>::get_static_coeffs();
                case 18: return GtinvDataVer2<3,18>::get_static_coeffs();
                case 19: return GtinvDataVer2<3,19>::get_static_coeffs();
                case 20: return GtinvDataVer2<3,20>::get_static_coeffs();
                default:
                    throw std::invalid_argument("Invalid L. (L3 > 20)");
            }

        //case 4: return GtinvData<4>::get_static_coeffs();
        //case 5: return GtinvData<5>::get_static_coeffs();
        //case 6: return GtinvData<6>::get_static_coeffs();
        default: throw std::invalid_argument("Invalid order");
    }
}


Readgtinv::Readgtinv(){}
Readgtinv::Readgtinv(
    const int gtinv_order,
    const vector1i& gtinv_maxl,
    const std::vector<bool>& gtinv_sym,
    const int n_type,
    const int version){

    if (version == 1){
        screening(gtinv_order, gtinv_maxl, gtinv_sym, n_type);
    }
    else {
        screening_v2(gtinv_order, gtinv_maxl, n_type);
    }
}

Readgtinv::~Readgtinv(){}


void Readgtinv::screening(
    const int gtinv_order,
    const vector1i& gtinv_maxl,
    const std::vector<bool>& gtinv_sym,
    const int n_type){

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

void Readgtinv::screening_v2(
    const int gtinv_order,
    const vector1i& gtinv_maxl,
    const int n_type){

    for (int order = 1; order <= gtinv_order; ++order){
        int maxl(0);
        if (order > 1) maxl = gtinv_maxl[order-2];

        for (int l1 = 0; l1 <= maxl; ++l1){
            const auto& l_array_all = get_l_array_v2(order, l1);
            const auto& m_array_all = get_m_array_v2(order, l1);
            const auto& coeffs_all = get_coeffs_v2(order, l1);

            for (size_t i = 0; i < l_array_all.size(); ++i){
                const vector1i& lcomb = l_array_all[i];
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
}


const vector3i& Readgtinv::get_lm_seq() const{ return lm_array; }
const vector2i& Readgtinv::get_l_comb() const{ return l_array; }
const vector2d& Readgtinv::get_lm_coeffs() const{ return coeffs; }
