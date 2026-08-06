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

    int sum_l = - std::accumulate(l_list.begin(), l_list.end(), 0);
    if (sum_l % 2 != 0){
        throw std::invalid_argument("Sum of angular numbers not even.");
    }
    pre = Precondition(l_list);
    const auto& row = pre.get_row();
    core_size = row.size();

    const int order = l_list.size();
    if (order == 2) order2(l_list);
    else if (order == 3) order3(l_list);
    else if (order == 4) order4(l_list);
    else if (order == 5) order5(l_list);
    else if (order == 6) order6(l_list);
}


void Projector::order2(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];

    auto& map_m_to_index2 = pre.get_map_m_to_index2();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    for (int m1=-l1; m1<=l1; ++m1){
        vector1i mv1 = {m1};
        int m2;
        bool nonzero1 = check_sum(mv1, l2, m2);
        if (!nonzero1)
            continue;
        int index = map_m_to_index2[m1];
        for (int m1p=-l1; m1p<=l1; ++m1p){
            vector1i mv2 = {m1p};
            int m2p;
            bool nonzero2 = check_sum(mv2, l2, m2p);
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

            core(index, index_p) = num;
            if (index != index_p){
                core(index_p, index) = num;
            }
        }
    }
}

void Projector::order3(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];

    auto& map_m_to_index3 = pre.get_map_m_to_index3();

    std::map<std::tuple<int, int>, double> cleb1;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        cleb1[{m1, m2}] = clebsch_gordan(l1, l2, l3, m1, m2, m1+m2);
    }

    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2){
        vector1i mv1 = {m1, m2};
        int m3;
        bool nonzero1 = check_sum(mv1, l3, m3);
        if (!nonzero1)
            continue;
        int index = map_m_to_index3[{m1, m2}];
        for (int m2p=-l2; m2p<=l2; ++m2p){
            vector1i mv2 = {m1p, m2p};
            int m3p;
            bool nonzero2 = check_sum(mv2, l3, m3p);
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

            core(index, index_p) = num;
            if (index != index_p){
                core(index_p, index) = num;
            }
        }
    }
}

void Projector::order4(const vector1i& l_list){
/***************************************************************

    Given l_list and (m1, m2, m3, m1p, m2p, m3p),
    the following quantity is calculated.

    double num(0.0);
    for (int l = abs(l1-l2); l < l1+l2+1; ++l){
        num += cleb[{l1, l2, l, m1, m2, -m3-m4}]
             * cleb[{l1, l2, l, m1p, m2p, -m3p-m4p}]
             * cleb[{l3, l, l4, m3, -m3-m4, -m4}]
             * cleb[{l3, l, l4, m3p, -m3p-m4p, -m4p}];
    num *= pow(-1, abs(m4-m4p))/(2*l4+1);

    The relationship
    C(l1, l2, l, m1, m2, m) = (-1)^(l1+l2-l) * C(l1, l2, l, -m1, -m2, -m)
    is used to reduce operations.

****************************************************************/

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];

    auto& map_m_to_index4 = pre.get_map_m_to_index4();

    map_tuple3_d cleb1, cleb2;
    for (int l = abs(l1-l2); l <= l1+l2; ++l)
    for (int m1=0; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        cleb1[{l, m1, m2}] = clebsch_gordan(l1, l2, l, m1, m2, m1+m2);
        for (int m3=-l3; m3<=l3; ++m3){
            cleb2[{l, m3, m1+m2}] = clebsch_gordan(l3, l, l4, m3, m1+m2, m1+m2+m3);
        }
    }

    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=0; m1<=l1; ++m1)
    for (int m1p=0; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p){
        vector1d prod_lq1;
        vector1i list_lq1;
        for (int l = abs(l1-l2); l < l1+l2+1; ++l){
            if (abs(m1+m2) > l)
                continue;
            double cg1 = cleb1[{l, m1, m2}];
            double cg2 = cleb1[{l, m1p, m2p}];
            prod_lq1.emplace_back(cg1 * cg2);
            list_lq1.emplace_back(l);
        }

        //int msum = m1 + m2;
        //int msump = m1p + m2p;
        //int lower = std::max(-l3, -l4 - msum);
        //int upper = std::min(l3, l4 - msum);
        //int lowerp = std::max(-l3, -l4 - msump);
        //int upperp = std::min(l3, l4 - msump);
        //for (int m3=lower; m3<=upper; ++m3){
        //    int m4 = - (m1 + m2 + m3);
        //    int index = map_m_to_index4[{m1, m2, m3}];
        //    int index2 = map_m_to_index4[{-m1, -m2, -m3}];
        //    for (int m3p=lowerp; m3<=upperp; ++m3p){
        //        int m4p = - (m1p + m2p + m3p);
        //        int index_p = map_m_to_index4[{m1p, m2p, m3p}];
        //        int index_p2 = map_m_to_index4[{-m1p, -m2p, -m3p}];
        //        if (index > index_p)
        //            continue;

        for (int m3=-l3; m3<=l3; ++m3){
            vector1i mv1 = {m1, m2, m3};
            int m4;
            if (!check_sum(mv1, l4, m4))
                continue;
            int index = map_m_to_index4[{m1, m2, m3}];
            int index2 = map_m_to_index4[{-m1, -m2, -m3}];

            for (int m3p=-l3; m3p<=l3; ++m3p){
                vector1i mv2 = {m1p, m2p, m3p};
                int m4p;
                if (!check_sum(mv2, l4, m4p))
                    continue;
                int index_p = map_m_to_index4[{m1p, m2p, m3p}];
                int index_p2 = map_m_to_index4[{-m1p, -m2p, -m3p}];
                if (index > index_p)
                    continue;

                double sign = ((abs(m4 - m4p) & 1) == 0) ? 1.0 : -1.0;
                double inv_norm = sign / (2*l4+1);

                double num(0.0);
                int cnt(0);
                for (auto l: list_lq1){
                    double prod1 = prod_lq1[cnt];
                    double cg3 = cleb2[{l, m3, m1+m2}];
                    double cg4 = cleb2[{l, m3p, m1p+m2p}];
                    num += prod1 * cg3 * cg4;
                    ++cnt;
                }
                num *= inv_norm;

                core(index, index_p) = num;
                core(index, index_p2) = num;
                core(index2, index_p) = num;
                core(index2, index_p2) = num;
                if (index != index_p){
                    core(index_p, index) = num;
                    core(index_p, index2) = num;
                    core(index_p2, index) = num;
                    core(index_p2, index2) = num;
                }
            }
        }
    }
}


void Projector::order5(const vector1i& l_list){
/*******************************************************************

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

**********************************************************************/

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];

    auto& map_m_to_index5 = pre.get_map_m_to_index5();

    std::map<std::tuple<int, int, int>, double> cleb1, cleb3;
    std::map<std::tuple<int, int, int, int>, double> cleb2;
    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1)
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        if (abs(m1+m2) > lq1)
            continue;
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            if (abs(m1+m2+m3) > lq2)
                continue;
            cleb2[{lq1,lq2,m3,m1+m2}]
                = clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3);
            for (int m4=-l4; m4<=l4; ++m4){
                int m5;
                if (!check_sum({m1, m2, m3, m4}, l5, m5))
                    continue;
                cleb3[{lq2,m4,m1+m2+m3}]
                    = clebsch_gordan(l4,lq2,l5,m4,m1+m2+m3,m1+m2+m3+m4);
            }
        }
    }

    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p){
        vector1d prod_lq1;
        vector1i list_lq1;
        //int M1 = abs(m1 + m2);
        //int M2 = abs(m1p + m2p);
        //int lq_min = std::max({abs(l1-l2), M1, M2});
        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
        //for (int lq1 = lq_min; lq1 < l1+l2+1; ++lq1){
            if (abs(m1+m2) > lq1)
                continue;
            if (abs(m1p+m2p) > lq1)
                continue;
            double cg1 = cleb1[{lq1,m1,m2}];
            double cg2 = cleb1[{lq1,m1p,m2p}];
            double prod = cg1 * cg2;
            prod_lq1.emplace_back(prod);
            list_lq1.emplace_back(lq1);
        }
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            vector1i list_lq2;
            int cnt(0);
            for (auto lq1: list_lq1){
                double prod1 = prod_lq1[cnt];
                for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                    if (abs(m1+m2+m3) > lq2)
                        continue;
                    if (abs(m1p+m2p+m3p) > lq2)
                        continue;
                    double cg3 = cleb2[{lq1,lq2,m3,m1+m2}];
                    double cg4 = cleb2[{lq1,lq2,m3p,m1p+m2p}];
                    double prod2 = prod1 * cg3 * cg4;
                    prod_lq2.emplace_back(prod2);
                    list_lq2.emplace_back(lq2);
                }
                ++cnt;
            }
            for (int m4=-l4; m4<=l4; ++m4){
                int m5;
                if (!check_sum({m1, m2, m3, m4}, l5, m5))
                    continue;
                int index = map_m_to_index5[{m1, m2, m3, m4}];
                for (int m4p=-l4; m4p<=l4; ++m4p){
                    int m5p;
                    if (!check_sum({m1p, m2p, m3p, m4p}, l5, m5p))
                        continue;
                    int index_p = map_m_to_index5[{m1p, m2p, m3p, m4p}];
                    if (index > index_p)
                        continue;

                    double sign = ((abs(m5 - m5p) & 1) == 0) ? 1.0 : -1.0;
                    double inv_norm = sign / (2*l5+1);

                    double num(0.0);
                    int cnt2(0);
                    for (auto lq2: list_lq2){
                        double prod2 = prod_lq2[cnt2];
                        double cg5 = cleb3[{lq2,m4,m1+m2+m3}];
                        double cg6 = cleb3[{lq2,m4p,m1p+m2p+m3p}];
                        num += prod2 * cg5 * cg6;
                        ++cnt2;
                    }
                    num *= inv_norm;

                    core(index, index_p) = num;
                    if (index != index_p){
                        core(index_p, index) = num;
                    }
                }
            }
        }
    }
}


void Projector::order6(const vector1i& l_list){
/*******************************************************************

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

**********************************************************************/

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];

    auto& map_m_to_index6 = pre.get_map_m_to_index6();

    std::map<std::tuple<int, int, int>, double> cleb1, cleb4;
    std::map<std::tuple<int, int, int, int>, double> cleb2, cleb3;
    for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1)
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        if (abs(m1+m2) > lq1)
            continue;
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            if (abs(m1+m2+m3) > lq2)
                continue;
            cleb2[{lq1,lq2,m3,m1+m2}]
                = clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3);
            for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3)
            for (int m4=-l4; m4<=l4; ++m4){
                if (abs(m1+m2+m3+m4) > lq3)
                    continue;
                cleb3[{lq2,lq3,m4,m1+m2+m3}]
                    = clebsch_gordan(l4,lq2,lq3,m4,m1+m2+m3,m1+m2+m3+m4);
                for (int m5=-l5; m5<=l5; ++m5){
                    int m6;
                    if (!check_sum({m1, m2, m3, m4, m5}, l6, m6))
                        continue;
                    cleb4[{lq3,m5,m1+m2+m3+m4}]
                        = clebsch_gordan(l5,lq3,l6,m5,m1+m2+m3+m4,m1+m2+m3+m4+m5);
                }
            }
        }
    }

    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p){
        vector1d prod_lq1;
        vector1i list_lq1;
        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
            if (abs(m1+m2) > lq1)
                continue;
            if (abs(m1p+m2p) > lq1)
                continue;
            double cg1 = cleb1[{lq1,m1,m2}];
            double cg2 = cleb1[{lq1,m1p,m2p}];
            double prod = cg1 * cg2;
            prod_lq1.emplace_back(prod);
            list_lq1.emplace_back(lq1);
        }
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            vector1i list_lq2;
            int cnt(0);
            for (auto lq1: list_lq1){
                double prod1 = prod_lq1[cnt];
                for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                    if (abs(m1+m2+m3) > lq2)
                        continue;
                    if (abs(m1p+m2p+m3p) > lq2)
                        continue;
                    double cg3 = cleb2[{lq1,lq2,m3,m1+m2}];
                    double cg4 = cleb2[{lq1,lq2,m3p,m1p+m2p}];
                    double prod2 = prod1 * cg3 * cg4;
                    prod_lq2.emplace_back(prod2);
                    list_lq2.emplace_back(lq2);
                }
                ++cnt;
            }
            for (int m4=-l4; m4<=l4; ++m4)
            for (int m4p=-l4; m4p<=l4; ++m4p){
                vector1d prod_lq3;
                vector1i list_lq3;
                int cnt2(0);
                for (auto lq2: list_lq2){
                    double prod2 = prod_lq2[cnt2];
                    for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3){
                        if (abs(m1+m2+m3+m4) > lq3)
                            continue;
                        if (abs(m1p+m2p+m3p+m4p) > lq3)
                            continue;
                        double cg5 = cleb3[{lq2,lq3,m4,m1+m2+m3}];
                        double cg6 = cleb3[{lq2,lq3,m4p,m1p+m2p+m3p}];
                        double prod3 = prod2 * cg5 * cg6;
                        prod_lq3.emplace_back(prod3);
                        list_lq3.emplace_back(lq3);
                    }
                    ++cnt2;
                }
                for (int m5=-l5; m5<=l5; ++m5){
                    vector1i mv1 = {m1, m2, m3, m4, m5};
                    int m6;
                    bool nonzero1 = check_sum(mv1, l6, m6);
                    if (!nonzero1)
                        continue;
                    int index = map_m_to_index6[{m1, m2, m3, m4, m5}];
                    for (int m5p=-l5; m5p<=l5; ++m5p){
                        vector1i mv2 = {m1p, m2p, m3p, m4p, m5p};
                        int m6p;
                        bool nonzero2 = check_sum(mv2, l6, m6p);
                        if (!nonzero2)
                            continue;
                        int index_p = map_m_to_index6[{m1p, m2p, m3p, m4p, m5p}];
                        if (index > index_p)
                            continue;

                        double sign = ((abs(m6 - m6p) & 1) == 0) ? 1.0 : -1.0;
                        double inv_norm = sign / (2*l6+1);

                        double num(0.0);
                        int cnt3(0);
                        for (auto lq3: list_lq3){
                            double prod3 = prod_lq3[cnt3];
                            double cg7 = cleb4[{lq3,m5,m1+m2+m3+m4}];
                            double cg8 = cleb4[{lq3,m5p,m1p+m2p+m3p+m4p}];
                            num += prod3 * cg7 * cg8;
                            ++cnt3;
                        }
                        num *= inv_norm;

                        core(index, index_p) = num;
                        if (index != index_p){
                            core(index_p, index) = num;
                        }
                    }
                }
            }
        }
    }
}

double Projector::clebsch_gordan
(const int& l1, const int& l2, const int& l,
 const int& m1, const int& m2, const int& m){

    return gsl_sf_coupling_3j(2*l1, 2*l2, 2*l, 2*m1, 2*m2, -2*m)
        * sqrt(2*l+1) * pow(-1, l1-l2+m);

}

Eigen::MatrixXd& Projector::get_core(){ return core; }
const vector1i& Projector::get_row() const{ return pre.get_row(); }
