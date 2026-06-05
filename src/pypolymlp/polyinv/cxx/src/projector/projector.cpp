/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

	    Main program for building projector for group-theoretic invariants

        # Quantum theory of angular momentum (Varshalovich, p.96)

*****************************************************************************/

#include "projector.h"

Projector::Projector(){}
Projector::~Projector(){}

std::mutex mtx;


void Projector::build_projector(const vector1i& l_list){

    const int order = l_list.size();
    if (order == 2) order2(l_list);
    else if (order == 3) order3(l_list);
    else if (order == 4) order4(l_list);
    else if (order == 5) order5(l_list);
    else if (order == 6) order6(l_list);
}


bool Projector::check_sum(const vector1i& l_list, const vector1i& m, int& mf){
    mf = - std::accumulate(m.begin(), m.end(), 0);
    if (abs(mf) > l_list[l_list.size()-1])
        return false;
    return true;
}


void Projector::precalc_common(
    const std::set<int>& nonzero_indices,
    std::map<int, int>& map_indices){

    const int core_size = nonzero_indices.size();
    row = vector1i(core_size);

    int seq = 0;
    for (int idx: nonzero_indices) {
        map_indices[idx] = seq;
        row[seq] = idx;
        ++seq;
    }
}


void Projector::order2_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];

    map_m_to_index2.clear();
    vector2i m_list;
    for (int m1=-l1; m1<=l1; ++m1){
        vector1i mv1 = {m1};
        int m2;
        if (check_sum(l_list, mv1, m2)){
            m_list.emplace_back(mv1);
            mv1.emplace_back(m2);
            map_m_to_index2[m1] = lm_to_matrix_index(l_list, mv1);
        }
    }

    std::set<int> nonzero_indices;
    for (const auto& mv1: m_list){
        int index = map_m_to_index2[mv1[0]];
        for (const auto& mv2: m_list){
            int index_p = map_m_to_index2[mv2[0]];
            if (index > index_p)
                continue;

            nonzero_indices.insert(index);
            nonzero_indices.insert(index_p);
        }
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order2(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];

    std::map<int, int> map_indices;
    order2_pre(l_list, map_indices);

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    for (int m1=-l1; m1<=l1; ++m1){
        vector1i mv1 = {m1};
        int m2;
        bool nonzero1 = check_sum(l_list, mv1, m2);
        if (!nonzero1)
            continue;
        int index = map_m_to_index2[m1];
        for (int m1p=-l1; m1p<=l1; ++m1p){
            vector1i mv2 = {m1p};
            int m2p;
            bool nonzero2 = check_sum(l_list, mv2, m2p);
            if (!nonzero2)
                continue;
            int index_p = map_m_to_index2[m1p];
            if (index > index_p)
                continue;

            double num;
            if (l1 == l2 and -m1 == m2 and -m1p == m2p){
                num = pow(-1, abs(m2 - m2p)) / (2 * l2 + 1);
            }
            else num = 0.0;

            core(map_indices[index], map_indices[index_p]) = num;
            if (index != index_p){
                core(map_indices[index_p], map_indices[index]) = num;
            }
        }
    }
}

void Projector::order3_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];

    map_m_to_index3.clear();
    vector2i m_list;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        vector1i mv1 = {m1, m2};
        int m3;
        if (check_sum(l_list, mv1, m3)){
            m_list.emplace_back(mv1);
            mv1.emplace_back(m3);
            map_m_to_index3[{m1, m2}] = lm_to_matrix_index(l_list, mv1);
        }
    }

    std::set<int> nonzero_indices;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (const auto& mv1: m_list){
        int index = map_m_to_index3[{mv1[0],mv1[1]}];
        for (const auto& mv2: m_list){
            int index_p = map_m_to_index3[{mv2[0],mv2[1]}];
            if (index > index_p)
                continue;

            std::lock_guard<std::mutex> lock(mtx);
            nonzero_indices.insert(index);
            nonzero_indices.insert(index_p);
        }
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order3(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];

    std::map<int, int> map_indices;
    order3_pre(l_list, map_indices);

    std::map<std::tuple<int, int>, double> cleb1;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        cleb1[{m1, m2}] = clebsch_gordan(l1, l2, l3, m1, m2, m1+m2);
    }

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2){
        vector1i mv1 = {m1, m2};
        int m3;
        bool nonzero1 = check_sum(l_list, mv1, m3);
        if (!nonzero1)
            continue;
        int index = map_m_to_index3[{m1, m2}];
        for (int m2p=-l2; m2p<=l2; ++m2p){
            vector1i mv2 = {m1p, m2p};
            int m3p;
            bool nonzero2 = check_sum(l_list, mv2, m3p);
            if (!nonzero2)
                continue;
            int index_p = map_m_to_index3[{m1p, m2p}];
            if (index > index_p)
                continue;

            double sign = ((abs(m3 - m3p) & 1) == 0) ? 1.0 : -1.0;
            double inv_norm = sign / (2*l3+1);

            double cg1 = cleb1[{m1, m2}];
            double cg2 = cleb1[{m1p, m2p}];
            double num = cg1 * cg2 * inv_norm;

            core(map_indices[index], map_indices[index_p]) = num;
            if (index != index_p){
                core(map_indices[index_p], map_indices[index]) = num;
            }
        }
    }
}

void Projector::order4_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];

    map_m_to_index4.clear();
    vector2i m_list;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3){
        vector1i mv1 = {m1, m2, m3};
        int m4;
        if (check_sum(l_list, mv1, m4)){
            m_list.emplace_back(mv1);
            mv1.emplace_back(m4);
            map_m_to_index4[{m1, m2, m3}] = lm_to_matrix_index(l_list, mv1);
        }
    }

    std::set<int> nonzero_indices;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (const auto& mv1: m_list){
        int index = map_m_to_index4[{mv1[0],mv1[1],mv1[2]}];
        for (const auto& mv2: m_list){
            int index_p = map_m_to_index4[{mv2[0],mv2[1],mv2[2]}];
            if (index > index_p)
                continue;

            std::lock_guard<std::mutex> lock(mtx);
            nonzero_indices.insert(index);
            nonzero_indices.insert(index_p);
        }
    }
    precalc_common(nonzero_indices, map_indices);
}

void Projector::order4(const vector1i& l_list){
/*
    Given l_list and (m1, m2, m3, m1p, m2p, m3p),
    the following quantity is calculated.

    double num(0.0);
    for (int l = abs(l1-l2); l < l1+l2+1; ++l){
        num += cleb[{l1, l2, l, m1, m2, -m3-m4}]
             * cleb[{l1, l2, l, m1p, m2p, -m3p-m4p}]
             * cleb[{l3, l, l4, m3, -m3-m4, -m4}]
             * cleb[{l3, l, l4, m3p, -m3p-m4p, -m4p}];
    num *= pow(-1, abs(m4-m4p))/(2*l4+1);
*/

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];

    std::map<int, int> map_indices;
    order4_pre(l_list, map_indices);

    std::map<std::tuple<int, int, int>, double> cleb1, cleb2;
    for (int l = abs(l1-l2); l < l1+l2+1; ++l)
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        cleb1[{l, m1, m2}] = clebsch_gordan(l1, l2, l, m1, m2, m1+m2);
        for (int m3=-l3; m3<=l3; ++m3){
            cleb2[{l, m3, m1+m2}] = clebsch_gordan(l3, l, l4, m3, m1+m2, m1+m2+m3);
        }
    }

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p){
        vector1d prod_lq1;
        for (int l = abs(l1-l2); l < l1+l2+1; ++l){
            double cg1 = cleb1[{l, m1, m2}];
            double cg2 = cleb1[{l, m1p, m2p}];
            prod_lq1.emplace_back(cg1 * cg2);
        }
        for (int m3=-l3; m3<=l3; ++m3){
            vector1i mv1 = {m1, m2, m3};
            int m4;
            bool nonzero1 = check_sum(l_list, mv1, m4);
            if (!nonzero1)
                continue;
            int index = map_m_to_index4[{m1, m2, m3}];
            for (int m3p=-l3; m3p<=l3; ++m3p){
                vector1i mv2 = {m1p, m2p, m3p};
                int m4p;
                bool nonzero2 = check_sum(l_list, mv2, m4p);
                if (!nonzero2)
                    continue;
                int index_p = map_m_to_index4[{m1p, m2p, m3p}];
                if (index > index_p)
                    continue;

                double sign = ((abs(m4 - m4p) & 1) == 0) ? 1.0 : -1.0;
                double inv_norm = sign / (2*l4+1);

                double num(0.0);
                int cnt(0);
                for (int l = abs(l1-l2); l < l1+l2+1; ++l){
                    double prod1 = prod_lq1[cnt];
                    double cg3 = cleb2[{l, m3, m1+m2}];
                    double cg4 = cleb2[{l, m3p, m1p+m2p}];
                    num += prod1 * cg3 * cg4;
                    ++cnt;
                }
                num *= inv_norm;

                core(map_indices[index], map_indices[index_p]) = num;
                if (index != index_p){
                    core(map_indices[index_p], map_indices[index]) = num;
                }
            }
        }
    }
}

void Projector::order5_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];

    map_m_to_index5.clear();
    vector2i m_list;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m4=-l4; m4<=l4; ++m4){
        vector1i mv1 = {m1, m2, m3, m4};
        int m5;
        if (check_sum(l_list, mv1, m5)){
            m_list.emplace_back(mv1);
            mv1.emplace_back(m5);
            map_m_to_index5[{m1, m2, m3, m4}] = lm_to_matrix_index(l_list, mv1);
        }
    }

    std::set<int> nonzero_indices;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (const auto& mv1: m_list){
        int index = map_m_to_index5[{mv1[0],mv1[1],mv1[2],mv1[3]}];
        for (const auto& mv2: m_list){
            int index_p = map_m_to_index5[{mv2[0],mv2[1],mv2[2],mv2[3]}];
            if (index > index_p)
                continue;

            std::lock_guard<std::mutex> lock(mtx);
            nonzero_indices.insert(index);
            nonzero_indices.insert(index_p);
        }
    }
    precalc_common(nonzero_indices, map_indices);
}

void Projector::order5(const vector1i& l_list){
/*
    Given l_list and (m1, m2, m3, m4, m1p, m2p, m3p, m4p),
    the following quantity is calculated.

    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
            num += cleb[{l1,l2,lq1,m1,m2,m1+m2}]
                 * cleb[{l1,l2,lq1,m1p,m2p,m1p+m2p}]
                 * cleb[{l3,lq1,lq2,m3,m1+m2,m1+m2+m3}]
                 * cleb[{l3,lq1,lq2,m3p,m1p+m2p,m1p+m2p+m3p}]
                 * cleb[{l4,lq2,l5,m4,m1+m2+m3,-m5}]
                 * cleb[{l4,lq2,l5,m4p,m1p+m2p+m3p,-m5p}];
        }
    }
    num *= pow(-1, abs(m5-m5p))/(2*l5+1);
*/
    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];

    std::map<int, int> map_indices;
    order5_pre(l_list, map_indices);

    std::map<std::tuple<int, int, int>, double> cleb1, cleb3;
    std::map<std::tuple<int, int, int, int>, double> cleb2;
    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1)
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            cleb2[{lq1,lq2,m3,m1+m2}]
                = clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3);
            for (int m4=-l4; m4<=l4; ++m4){
                cleb3[{lq2,m4,m1+m2+m3}]
                    = clebsch_gordan(l4,lq2,l5,m4,m1+m2+m3,m1+m2+m3+m4);
            }
        }
    }

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p){
        vector1d prod_lq1;
        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
            double cg1 = cleb1[{lq1,m1,m2}];
            double cg2 = cleb1[{lq1,m1p,m2p}];
            double prod = cg1 * cg2;
            prod_lq1.emplace_back(prod);
        }
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            int cnt(0);
            for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
                double prod1 = prod_lq1[cnt];
                for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                    double cg3 = cleb2[{lq1,lq2,m3,m1+m2}];
                    double cg4 = cleb2[{lq1,lq2,m3p,m1p+m2p}];
                    double prod2 = prod1 * cg3 * cg4;
                    prod_lq2.emplace_back(prod2);
                }
                ++cnt;
            }
            for (int m4=-l4; m4<=l4; ++m4){
                vector1i mv1 = {m1, m2, m3, m4};
                int m5;
                bool nonzero1 = check_sum(l_list, mv1, m5);
                if (!nonzero1)
                    continue;
                int index = map_m_to_index5[{m1, m2, m3, m4}];
                for (int m4p=-l4; m4p<=l4; ++m4p){
                    vector1i mv2 = {m1p, m2p, m3p, m4p};
                    int m5p;
                    bool nonzero2 = check_sum(l_list, mv2, m5p);
                    if (!nonzero2)
                        continue;
                    int index_p = map_m_to_index5[{m1p, m2p, m3p, m4p}];
                    if (index > index_p)
                        continue;

                    double sign = ((abs(m5 - m5p) & 1) == 0) ? 1.0 : -1.0;
                    double inv_norm = sign / (2*l5+1);

                    double num(0.0);
                    int cnt2(0);
                    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
                        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                            double prod2 = prod_lq2[cnt2];
                            double cg5 = cleb3[{lq2,m4,m1+m2+m3}];
                            double cg6 = cleb3[{lq2,m4p,m1p+m2p+m3p}];
                            num += prod2 * cg5 * cg6;
                            ++cnt2;
                        }
                    }
                    num *= inv_norm;

                    core(map_indices[index], map_indices[index_p]) = num;
                    if (index != index_p){
                        core(map_indices[index_p], map_indices[index]) = num;
                    }
                }
            }
        }
    }
}


void Projector::order6_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];

    map_m_to_index6.clear();
    vector2i m_list;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m5=-l5; m5<=l5; ++m5){
        vector1i mv1 = {m1, m2, m3, m4, m5};
        int m6;
        if (check_sum(l_list, mv1, m6)){
            m_list.emplace_back(mv1);
            mv1.emplace_back(m6);
            map_m_to_index6[{m1, m2, m3, m4 ,m5}] = lm_to_matrix_index(l_list, mv1);
        }
    }

    std::set<int> nonzero_indices;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (const auto& mv1: m_list){
        int index = map_m_to_index6[{mv1[0],mv1[1],mv1[2],mv1[3],mv1[4]}];
        for (const auto& mv2: m_list){
            int index_p = map_m_to_index6[{mv2[0],mv2[1],mv2[2],mv2[3],mv2[4]}];
            if (index > index_p)
                continue;

            std::lock_guard<std::mutex> lock(mtx);
            nonzero_indices.insert(index);
            nonzero_indices.insert(index_p);
        }
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order6(const vector1i& l_list){
/*
    Given l_list and (m1, m2, m3, m4, m5, m1p, m2p, m3p, m4p, m5p),
    the following quantity is calculated.

    double num(0);
    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
            for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3){
                num += clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2)
                    * clebsch_gordan(l1,l2,lq1,m1p,m2p,m1p+m2p)
                    * clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3)
                    * clebsch_gordan(l3,lq1,lq2,m3p,m1p+m2p,m1p+m2p+m3p)
                    * clebsch_gordan(l4,lq2,lq3,m4,m1+m2+m3,m1+m2+m3+m4)
                    * clebsch_gordan
                        (l4,lq2,lq3,m4p,m1p+m2p+m3p,m1p+m2p+m3p+m4p)
                    * clebsch_gordan(l5,lq3,l6,m5,m1+m2+m3+m4,-m6)
                    * clebsch_gordan(l5,lq3,l6,m5p,m1p+m2p+m3p+m4p,-m6p);
            }
        }
    }
    num *= pow(-1, abs(m6-m6p))/(2*l6+1);
*/

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];

    std::map<int, int> map_indices;
    order6_pre(l_list, map_indices);

    std::map<std::tuple<int, int, int>, double> cleb1, cleb4;
    std::map<std::tuple<int, int, int, int>, double> cleb2, cleb3;
    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1)
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            cleb2[{lq1,lq2,m3,m1+m2}]
                = clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3);
            for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3)
            for (int m4=-l4; m4<=l4; ++m4){
                cleb3[{lq2,lq3,m4,m1+m2+m3}]
                    = clebsch_gordan(l4,lq2,lq3,m4,m1+m2+m3,m1+m2+m3+m4);
                for (int m5=-l5; m5<=l5; ++m5){
                    cleb4[{lq3,m5,m1+m2+m3+m4}]
                        = clebsch_gordan(l5,lq3,l6,m5,m1+m2+m3+m4,m1+m2+m3+m4+m5);
                }
            }
        }
    }

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p){
        vector1d prod_lq1;
        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
            double cg1 = cleb1[{lq1,m1,m2}];
            double cg2 = cleb1[{lq1,m1p,m2p}];
            double prod = cg1 * cg2;
            prod_lq1.emplace_back(prod);
        }
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            int cnt(0);
            for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
                double prod1 = prod_lq1[cnt];
                for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                    double cg3 = cleb2[{lq1,lq2,m3,m1+m2}];
                    double cg4 = cleb2[{lq1,lq2,m3p,m1p+m2p}];
                    double prod2 = prod1 * cg3 * cg4;
                    prod_lq2.emplace_back(prod2);
                }
                ++cnt;
            }
            for (int m4=-l4; m4<=l4; ++m4)
            for (int m4p=-l4; m4p<=l4; ++m4p){
                vector1d prod_lq3;
                int cnt2(0);
                for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
                    for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                        double prod2 = prod_lq2[cnt2];
                        for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3){
                            double cg5 = cleb3[{lq2,lq3,m4,m1+m2+m3}];
                            double cg6 = cleb3[{lq2,lq3,m4p,m1p+m2p+m3p}];
                            double prod3 = prod2 * cg5 * cg6;
                            prod_lq3.emplace_back(prod3);
                        }
                        ++cnt2;
                    }
                }

                for (int m5=-l5; m5<=l5; ++m5){
                    vector1i mv1 = {m1, m2, m3, m4, m5};
                    int m6;
                    bool nonzero1 = check_sum(l_list, mv1, m6);
                    if (!nonzero1)
                        continue;
                    int index = map_m_to_index6[{m1, m2, m3, m4, m5}];
                    for (int m5p=-l5; m5p<=l5; ++m5p){
                        vector1i mv2 = {m1p, m2p, m3p, m4p, m5p};
                        int m6p;
                        bool nonzero2 = check_sum(l_list, mv2, m6p);
                        if (!nonzero2)
                            continue;
                        int index_p = map_m_to_index6[{m1p, m2p, m3p, m4p, m5p}];
                        if (index > index_p)
                            continue;

                        double sign = ((abs(m6 - m6p) & 1) == 0) ? 1.0 : -1.0;
                        double inv_norm = sign / (2*l6+1);

                        double num(0.0);
                        int cnt3(0);
                        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
                            for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                                for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3){
                                    double prod3 = prod_lq3[cnt3];
                                    double cg7 = cleb4[{lq3,m5,m1+m2+m3+m4}];
                                    double cg8 = cleb4[{lq3,m5p,m1p+m2p+m3p+m4p}];
                                    num += prod3 * cg7 * cg8;
                                    ++cnt3;
                                }
                            }
                        }
                        num *= inv_norm;

                        core(map_indices[index], map_indices[index_p]) = num;
                        if (index != index_p){
                            core(map_indices[index_p], map_indices[index]) = num;
                        }
                    }
                }
            }
        }
    }
/*
        double num(0);
        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
            for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3){
                    num += clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2)
                        * clebsch_gordan(l1,l2,lq1,m1p,m2p,m1p+m2p)
                        * clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3)
                        * clebsch_gordan(l3,lq1,lq2,m3p,m1p+m2p,m1p+m2p+m3p)
                        * clebsch_gordan(l4,lq2,lq3,m4,m1+m2+m3,m1+m2+m3+m4)
                        * clebsch_gordan
                            (l4,lq2,lq3,m4p,m1p+m2p+m3p,m1p+m2p+m3p+m4p)
                        * clebsch_gordan(l5,lq3,l6,m5,m1+m2+m3+m4,-m6)
                        * clebsch_gordan(l5,lq3,l6,m5p,m1p+m2p+m3p+m4p,-m6p);
                }
            }
        }
        num *= pow(-1, abs(m6-m6p))/(2*l6+1);
    */
}

int Projector::lm_to_matrix_index(const vector1i& l_list, const vector1i& m_array) {
/*
    int Projector::lm_to_matrix_index
    (const vector1i& l_list, const vector1i& m_array){

        vector1i lpm_list(l_list.size()), l_list2(l_list.size());
        for (int i = 0; i < l_list.size(); ++i){
            lpm_list[i] = m_array[i] + l_list[i];
            l_list2[i] = 2 * l_list[i] + 1;
        }

        int index(0);
        for (int i = 0; i < lpm_list.size(); ++i){
            int tmp(lpm_list[i]);
            for (int j = i+1; j < l_list2.size(); ++j){
                tmp *= l_list2[j];
            }
            index += tmp;
        }
        return index;
    }
*/
    int index = 0;
    long long multiplier = 1;
    int size = l_list.size();

    for (int i = size - 1; i >= 0; --i) {
        index += (m_array[i] + l_list[i]) * multiplier;
        multiplier *= (2 * l_list[i] + 1);
    }

    return index;
}


double Projector::clebsch_gordan
(const int& l1, const int& l2, const int& l,
 const int& m1, const int& m2, const int& m){

    return gsl_sf_coupling_3j(2*l1, 2*l2, 2*l, 2*m1, 2*m2, -2*m)
        * sqrt(2*l+1) * pow(-1, l1-l2+m);

}

Eigen::MatrixXd& Projector::get_core(){ return core; }
const vector1i& Projector::get_row() const{ return row; }
