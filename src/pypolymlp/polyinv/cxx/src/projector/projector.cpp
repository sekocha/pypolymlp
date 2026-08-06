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
    else if (order == 7) order7(l_list);
}


void Projector::order2(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];

    auto& map_m = pre.get_map_m_to_index2();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    for (int m1=-l1; m1<=l1; ++m1){
        int m2;
        bool nonzero1 = check_sum({m1}, l2, m2);
        if (!nonzero1)
            continue;
        int index = map_m[m1];
        for (int m1p=-l1; m1p<=l1; ++m1p){
            int m2p;
            bool nonzero2 = check_sum({m1p}, l2, m2p);
            if (!nonzero2)
                continue;
            int index_p = map_m[m1p];
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

    map_tuple2_d cleb1;
    for (int m1=0; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        if (abs(m1+m2) > l3)
            continue;
        cleb1[{m1, m2}] = clebsch_gordan(l1, l2, l3, m1, m2, m1+m2);
    }

    auto& map_m = pre.get_map_m_to_index3();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int m1=0; m1<=l1; ++m1)
    for (int m1p=0; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2){
        int m3;
        bool nonzero1 = check_sum({m1, m2}, l3, m3);
        if (!nonzero1)
            continue;
        int index = map_m[{m1, m2}];
        int index2 = map_m[{-m1, -m2}];
        for (int m2p=-l2; m2p<=l2; ++m2p){
            int m3p;
            bool nonzero2 = check_sum({m1p, m2p}, l3, m3p);
            if (!nonzero2)
                continue;
            int index_p = map_m[{m1p, m2p}];
            int index_p2 = map_m[{-m1p, -m2p}];
            if (index > index_p)
                continue;

            double sign = ((abs(m3 - m3p) & 1) == 0) ? 1.0 : -1.0;
            double inv_norm = sign / (2*l3+1);

            double cg1 = cleb1[{m1, m2}];
            double cg2 = cleb1[{m1p, m2p}];
            double num = cg1 * cg2 * inv_norm;
            assign_core(core, num, index, index_p, index2, index_p2);
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

    map_tuple3_d cleb1, cleb2;
    for (int l = abs(l1-l2); l <= l1+l2; ++l)
    for (int m1=0; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        int sum2 = m1 + m2;
        if (abs(sum2) > l)
            continue;
        cleb1[{l, m1, m2}] = clebsch_gordan(l1, l2, l, m1, m2, sum2);
        for (int m3=-l3; m3<=l3; ++m3){
            int sum3 = sum2 + m3;
            if (abs(sum3) > l4)
                continue;
            cleb2[{l, m3, sum2}] = clebsch_gordan(l3, l, l4, m3, sum2, sum3);
        }
    }

    auto& map_m = pre.get_map_m_to_index4();
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
        set_inter_prod_first(cleb1, l1, l2, {m1, m2}, {m1p, m2p}, prod_lq1, list_lq1);
        for (int m3=-l3; m3<=l3; ++m3){
            int m4;
            if (!check_sum({m1, m2, m3}, l4, m4))
                continue;
            int index = map_m[{m1, m2, m3}];
            int index2 = map_m[{-m1, -m2, -m3}];
            for (int m3p=-l3; m3p<=l3; ++m3p){
                int m4p;
                if (!check_sum({m1p, m2p, m3p}, l4, m4p))
                    continue;
                int index_p = map_m[{m1p, m2p, m3p}];
                int index_p2 = map_m[{-m1p, -m2p, -m3p}];
                if (index > index_p)
                    continue;

                double sign = ((abs(m4 - m4p) & 1) == 0) ? 1.0 : -1.0;
                double inv_norm = sign / (2*l4+1);
                double num = set_final_prod(
                    prod_lq1, list_lq1, cleb2, {m1, m2, m3}, {m1p, m2p, m3p}
                );
                num *= inv_norm;
                assign_core(core, num, index, index_p, index2, index_p2);
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

    map_tuple3_d cleb1, cleb3;
    map_tuple4_d cleb2;
    for (int lq1 = abs(l1-l2); lq1 <= l1+l2; ++lq1)
    for (int m1=0; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        int sum2 = m1 + m2;
        if (abs(sum2) > lq1)
            continue;
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,sum2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            int sum3 = sum2 + m3;
            if (abs(sum3) > lq2)
                continue;
            cleb2[{lq1,lq2,m3,sum2}] = clebsch_gordan(l3,lq1,lq2,m3,sum2,sum3);
            for (int m4=-l4; m4<=l4; ++m4){
                int sum4 = sum3 + m4;
                if (abs(sum4) > l5)
                    continue;
                cleb3[{lq2,m4,sum3}] = clebsch_gordan(l4,lq2,l5,m4,sum3,sum4);
            }
        }
    }

    auto& map_m = pre.get_map_m_to_index5();
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
        set_inter_prod_first(cleb1, l1, l2, {m1, m2}, {m1p, m2p}, prod_lq1, list_lq1);
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            vector1i list_lq2;
            set_inter_prod(
                prod_lq1, list_lq1, cleb2, l3,
                {m1, m2, m3}, {m1p, m2p, m3p},
                prod_lq2, list_lq2);
            for (int m4=-l4; m4<=l4; ++m4){
                int m5;
                if (!check_sum({m1, m2, m3, m4}, l5, m5))
                    continue;
                int index = map_m[{m1, m2, m3, m4}];
                int index2 = map_m[{-m1, -m2, -m3, -m4}];
                for (int m4p=-l4; m4p<=l4; ++m4p){
                    int m5p;
                    if (!check_sum({m1p, m2p, m3p, m4p}, l5, m5p))
                        continue;
                    int index_p = map_m[{m1p, m2p, m3p, m4p}];
                    int index_p2 = map_m[{-m1p, -m2p, -m3p, -m4p}];
                    if (index > index_p)
                        continue;

                    double sign = ((abs(m5 - m5p) & 1) == 0) ? 1.0 : -1.0;
                    double inv_norm = sign / (2*l5+1);
                    double num = set_final_prod(
                        prod_lq2, list_lq2, cleb3,
                        {m1, m2, m3, m4}, {m1p, m2p, m3p, m4p});
                    num *= inv_norm;
                    assign_core(core, num, index, index_p, index2, index_p2);
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

    map_tuple3_d cleb1, cleb4;
    map_tuple4_d cleb2, cleb3;
    for (int lq1 = abs(l1-l2); lq1 <= l1+l2; ++lq1)
    for (int m1=0; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        int sum2 = m1 + m2;
        if (abs(sum2) > lq1)
            continue;
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,sum2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            int sum3 = sum2 + m3;
            if (abs(sum3) > lq2)
                continue;
            cleb2[{lq1,lq2,m3,sum2}] = clebsch_gordan(l3,lq1,lq2,m3,sum2,sum3);
            for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3)
            for (int m4=-l4; m4<=l4; ++m4){
                int sum4 = sum3 + m4;
                if (abs(sum4) > lq3)
                    continue;
                cleb3[{lq2,lq3,m4,sum3}] = clebsch_gordan(l4,lq2,lq3,m4,sum3,sum4);
                for (int m5=-l5; m5<=l5; ++m5){
                    int sum5 = sum4 + m5;
                    if (abs(sum5) > l6)
                        continue;
                    cleb4[{lq3,m5,sum4}] = clebsch_gordan(l5,lq3,l6,m5,sum4,sum5);
                }
            }
        }
    }

    auto& map_m = pre.get_map_m_to_index6();
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
        set_inter_prod_first(cleb1, l1, l2, {m1, m2}, {m1p, m2p}, prod_lq1, list_lq1);
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            vector1i list_lq2;
            set_inter_prod(
                prod_lq1, list_lq1, cleb2, l3,
                {m1, m2, m3}, {m1p, m2p, m3p},
                prod_lq2, list_lq2);
            for (int m4=-l4; m4<=l4; ++m4)
            for (int m4p=-l4; m4p<=l4; ++m4p){
                vector1d prod_lq3;
                vector1i list_lq3;
                set_inter_prod(
                    prod_lq2, list_lq2, cleb3, l4,
                    {m1, m2, m3, m4}, {m1p, m2p, m3p, m4p},
                    prod_lq3, list_lq3);
                for (int m5=-l5; m5<=l5; ++m5){
                    vector1i mv1 = {m1, m2, m3, m4, m5};
                    int m6;
                    bool nonzero1 = check_sum(mv1, l6, m6);
                    if (!nonzero1)
                        continue;
                    int index = map_m[{m1, m2, m3, m4, m5}];
                    int index2 = map_m[{-m1, -m2, -m3, -m4, -m5}];
                    for (int m5p=-l5; m5p<=l5; ++m5p){
                        vector1i mv2 = {m1p, m2p, m3p, m4p, m5p};
                        int m6p;
                        bool nonzero2 = check_sum(mv2, l6, m6p);
                        if (!nonzero2)
                            continue;
                        int index_p = map_m[{m1p, m2p, m3p, m4p, m5p}];
                        int index_p2 = map_m[{-m1p, -m2p, -m3p, -m4p, -m5p}];
                        if (index > index_p)
                            continue;

                        double sign = ((abs(m6 - m6p) & 1) == 0) ? 1.0 : -1.0;
                        double inv_norm = sign / (2*l6+1);
                        double num = set_final_prod(
                            prod_lq3, list_lq3, cleb4,
                            {m1, m2, m3, m4, m5}, {m1p, m2p, m3p, m4p, m5p});
                        num *= inv_norm;
                        assign_core(core, num, index, index_p, index2, index_p2);
                    }
                }
            }
        }
    }
}

void Projector::order7(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];
    const int l7 = l_list[6];

    map_tuple3_d cleb1, cleb5;
    map_tuple4_d cleb2, cleb3, cleb4;
    for (int lq1 = abs(l1-l2); lq1 <= l1+l2; ++lq1)
    for (int m1=0; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        int sum2 = m1 + m2;
        if (abs(sum2) > lq1)
            continue;
        cleb1[{lq1,m1,m2}] = clebsch_gordan(l1,l2,lq1,m1,m2,sum2);
        for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2)
        for (int m3=-l3; m3<=l3; ++m3){
            int sum3 = sum2 + m3;
            if (abs(sum3) > lq2)
                continue;
            cleb2[{lq1,lq2,m3,sum2}] = clebsch_gordan(l3,lq1,lq2,m3,sum2,sum3);
            for (int lq3 = abs(l4-lq2); lq3 < l4+lq2+1; ++lq3)
            for (int m4=-l4; m4<=l4; ++m4){
                int sum4 = sum3 + m4;
                if (abs(sum4) > lq3)
                    continue;
                cleb3[{lq2,lq3,m4,sum3}] = clebsch_gordan(l4,lq2,lq3,m4,sum3,sum4);
                for (int lq4 = abs(l5-lq3); lq4 < l5+lq3+1; ++lq4)
                for (int m5=-l5; m5<=l5; ++m5){
                    int sum5 = sum4 + m5;
                    if (abs(sum5) > lq4)
                        continue;
                    cleb4[{lq3,lq4,m5,sum4}] = clebsch_gordan(l5,lq3,lq4,m5,sum4,sum5);
                    for (int m6=-l6; m6<=l6; ++m6){
                        int sum6 = sum5 + m6;
                        if (abs(sum6) > l7)
                            continue;
                        cleb5[{lq4,m6,sum5}] = clebsch_gordan(l6,lq4,l7,m6,sum5,sum6);
                    }
                }
            }
        }
    }

    auto& map_m = pre.get_map_m_to_index7();
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
        set_inter_prod_first(cleb1, l1, l2, {m1, m2}, {m1p, m2p}, prod_lq1, list_lq1);
        for (int m3=-l3; m3<=l3; ++m3)
        for (int m3p=-l3; m3p<=l3; ++m3p){
            vector1d prod_lq2;
            vector1i list_lq2;
            set_inter_prod(prod_lq1, list_lq1, cleb2, l3,
                {m1, m2, m3}, {m1p, m2p, m3p},
                prod_lq2, list_lq2);
            for (int m4=-l4; m4<=l4; ++m4)
            for (int m4p=-l4; m4p<=l4; ++m4p){
                vector1d prod_lq3;
                vector1i list_lq3;
                set_inter_prod(prod_lq2, list_lq2, cleb3, l4,
                    {m1, m2, m3, m4}, {m1p, m2p, m3p, m4p},
                    prod_lq3, list_lq3);
                for (int m5=-l5; m5<=l5; ++m5)
                for (int m5p=-l5; m5p<=l5; ++m5p){
                    vector1d prod_lq4;
                    vector1i list_lq4;
                    set_inter_prod(prod_lq3, list_lq3, cleb4, l5,
                        {m1, m2, m3, m4, m5}, {m1p, m2p, m3p, m4p, m5p},
                        prod_lq4, list_lq4);
                    for (int m6=-l6; m6<=l6; ++m6){
                        vector1i mv1 = {m1, m2, m3, m4, m5, m6};
                        int m7;
                        bool nonzero1 = check_sum(mv1, l7, m7);
                        if (!nonzero1)
                            continue;
                        int index = map_m[{m1, m2, m3, m4, m5, m6}];
                        int index2 = map_m[{-m1, -m2, -m3, -m4, -m5, -m6}];
                        for (int m6p=-l6; m6p<=l6; ++m6p){
                            vector1i mv2 = {m1p, m2p, m3p, m4p, m5p, m6p};
                            int m7p;
                            bool nonzero2 = check_sum(mv2, l7, m7p);
                            if (!nonzero2)
                                continue;
                            int index_p = map_m[{m1p, m2p, m3p, m4p, m5p, m6p}];
                            int index_p2 = map_m[{-m1p, -m2p, -m3p, -m4p, -m5p, -m6p}];
                            if (index > index_p)
                                continue;

                            double sign = ((abs(m7 - m7p) & 1) == 0) ? 1.0 : -1.0;
                            double inv_norm = sign / (2*l7+1);
                            double num = set_final_prod(
                                prod_lq4, list_lq4, cleb5,
                                {m1, m2, m3, m4, m5, m6},
                                {m1p, m2p, m3p, m4p, m5p, m6p});
                            num *= inv_norm;
                            assign_core(core, num, index, index_p, index2, index_p2);
                        }
                    }
                }
            }
        }
    }
}

void Projector::set_inter_prod_first(
    map_tuple3_d& cleb,
    const int l1,
    const int l2,
    const vector1i& list_m,
    const vector1i& list_mp,
    vector1d& prod_lq,
    vector1i& list_lq
){
    int sum_m = std::accumulate(list_m.begin(), list_m.end(), 0);
    int sum_mp = std::accumulate(list_mp.begin(), list_mp.end(), 0);
    for (int lq1 = abs(l1-l2); lq1 <= l1 + l2; ++lq1){
        if (abs(sum_m) > lq1 or abs(sum_mp) > lq1)
            continue;
        double cg1 = cleb[{lq1, list_m[0], list_m[1]}];
        double cg2 = cleb[{lq1, list_mp[0], list_mp[1]}];
        double prod = cg1 * cg2;
        prod_lq.emplace_back(prod);
        list_lq.emplace_back(lq1);
    }
}

void Projector::set_inter_prod(
    const vector1d& prod_lq_prev,
    const vector1i& list_lq_prev,
    map_tuple4_d& cleb,
    const int l,
    const vector1i& list_m,
    const vector1i& list_mp,
    vector1d& prod_lq,
    vector1i& list_lq
){
    int sum_m = std::accumulate(list_m.begin(), list_m.end(), 0);
    int sum_mp = std::accumulate(list_mp.begin(), list_mp.end(), 0);
    int cnt(0);
    for (auto lq1: list_lq_prev){
        double prod1 = prod_lq_prev[cnt];
        for (int lq2 = abs(l - lq1); lq2 <= l + lq1; ++lq2){
            if (abs(sum_m) > lq2 or abs(sum_mp) > lq2)
                continue;

            int m_end = *(list_m.end()-1);
            int mp_end = *(list_mp.end()-1);
            double cg1 = cleb[{lq1, lq2, m_end, sum_m - m_end}];
            double cg2 = cleb[{lq1, lq2, mp_end, sum_mp - mp_end}];
            double prod2 = prod1 * cg1 * cg2;
            prod_lq.emplace_back(prod2);
            list_lq.emplace_back(lq2);
        }
        ++cnt;
    }

}

double Projector::set_final_prod(
    const vector1d& prod_lq_prev,
    const vector1i& list_lq_prev,
    map_tuple3_d& cleb,
    const vector1i& list_m,
    const vector1i& list_mp
){
    int sum_m = std::accumulate(list_m.begin(), list_m.end()-1, 0);
    int sum_mp = std::accumulate(list_mp.begin(), list_mp.end()-1, 0);
    int m_end = *(list_m.end()-1);
    int mp_end = *(list_mp.end()-1);

    double num(0.0);
    int cnt(0);
    for (auto l: list_lq_prev){
        double prod1 = prod_lq_prev[cnt];
        double cg1 = cleb[{l, m_end, sum_m}];
        double cg2 = cleb[{l, mp_end, sum_mp}];
        num += prod1 * cg1 * cg2;
        ++cnt;
    }
    return num;
}

void Projector::assign_core(
    Eigen::MatrixXd& core,
    const double num,
    const int index,
    const int index_p,
    const int index2,
    const int index_p2
){
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


double Projector::clebsch_gordan
(const int& l1, const int& l2, const int& l,
 const int& m1, const int& m2, const int& m){

    return gsl_sf_coupling_3j(2*l1, 2*l2, 2*l, 2*m1, 2*m2, -2*m)
        * sqrt(2*l+1) * pow(-1, l1-l2+m);

}

Eigen::MatrixXd& Projector::get_core(){ return core; }
const vector1i& Projector::get_row() const{ return pre.get_row(); }
