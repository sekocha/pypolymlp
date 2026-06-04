/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

	    Main program for building projector for group-theoretic invariants

        # Quantum theory of angular momentum (Varshalovich, p.96)

*****************************************************************************/

#include "projector.h"

Projector::Projector(){}
Projector::~Projector(){}


void Projector::build_projector(const vector1i& l_list){

    const int order = l_list.size();
    if (order == 2) order2(l_list);
    else if (order == 3) order3(l_list);
    else if (order == 4) order4(l_list);
    else if (order == 5) order5(l_list);
    else if (order == 6) order6(l_list);
}


bool Projector::check_m_nonzero(
    const vector1i& l_list,
    vector1i& mv1,
    vector1i& mv2,
    int& index,
    int& index_p){

    int mf = - std::accumulate(mv1.begin(), mv1.end(), 0);
    if (abs(mf) > l_list[l_list.size()-1])
        return false;

    int mfp = - std::accumulate(mv2.begin(), mv2.end(), 0);
    if (abs(mfp) > l_list[l_list.size()-1])
        return false;

    mv1.emplace_back(mf);
    mv2.emplace_back(mfp);
    index = lm_to_matrix_index(l_list, mv1);
    index_p = lm_to_matrix_index(l_list, mv2);
    if (index > index_p)
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
    const int l2 = l_list[1];

    std::set<int> nonzero_indices;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p){
        int index, index_p;
        vector1i mv1 = {m1};
        vector1i mv2 = {m1p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        nonzero_indices.insert(index);
        nonzero_indices.insert(index_p);
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

    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p){
        int index, index_p;
        vector1i mv1 = {m1};
        vector1i mv2 = {m1p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        int m2 = mv1[1], m2p = mv2[1];
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

void Projector::order3_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];

    std::set<int> nonzero_indices;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2};
        vector1i mv2 = {m1p, m2p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        nonzero_indices.insert(index);
        nonzero_indices.insert(index_p);
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order3(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];

    std::map<int, int> map_indices;
    order3_pre(l_list, map_indices);

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(4) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2};
        vector1i mv2 = {m1p, m2p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        int m3 = mv1[2], m3p = mv2[2];
        double num = pow(-1, abs(m3-m3p))/(2*l3+1)
              * clebsch_gordan(l1, l2, l3, m1, m2, -m3)
              * clebsch_gordan(l1, l2, l3, m1p, m2p, -m3p);

        core(map_indices[index], map_indices[index_p]) = num;
        if (index != index_p){
            core(map_indices[index_p], map_indices[index]) = num;
        }
    }
}

void Projector::order4_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];

    std::set<int> nonzero_indices;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m3p=-l3; m3p<=l3; ++m3p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2, m3};
        vector1i mv2 = {m1p, m2p, m3p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        nonzero_indices.insert(index);
        nonzero_indices.insert(index_p);
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order4(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];

    std::map<int, int> map_indices;
    order4_pre(l_list, map_indices);

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(6) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m3p=-l3; m3p<=l3; ++m3p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2, m3};
        vector1i mv2 = {m1p, m2p, m3p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        int m4 = mv1[3], m4p = mv2[3];
        double num(0);
        for (int l = abs(l1-l2); l < l1+l2+1; ++l){
            num += clebsch_gordan(l1, l2, l, m1, m2, -m3-m4)
                * clebsch_gordan(l1, l2, l, m1p, m2p, -m3p-m4p)
                * clebsch_gordan(l3, l, l4, m3, -m3-m4, -m4)
                * clebsch_gordan(l3, l, l4, m3p, -m3p-m4p, -m4p);
        }
        num *= pow(-1, abs(m4-m4p))/(2*l4+1);

        core(map_indices[index], map_indices[index_p]) = num;
        if (index != index_p){
            core(map_indices[index_p], map_indices[index]) = num;
        }
    }
}

void Projector::order5_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];

    std::set<int> nonzero_indices;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m3p=-l3; m3p<=l3; ++m3p)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m4p=-l4; m4p<=l4; ++m4p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2, m3, m4};
        vector1i mv2 = {m1p, m2p, m3p, m4p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        nonzero_indices.insert(index);
        nonzero_indices.insert(index_p);
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order5(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];

    std::map<int, int> map_indices;
    order5_pre(l_list, map_indices);

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(8) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m3p=-l3; m3p<=l3; ++m3p)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m4p=-l4; m4p<=l4; ++m4p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2, m3, m4};
        vector1i mv2 = {m1p, m2p, m3p, m4p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        int m5 = mv1[4], m5p = mv2[4];
        double num(0);
        for (int lq1 = abs(l1-l2); lq1 < l1+l2+1; ++lq1){
            for (int lq2 = abs(l3-lq1); lq2 < l3+lq1+1; ++lq2){
                num += clebsch_gordan(l1,l2,lq1,m1,m2,m1+m2)
                    * clebsch_gordan(l1,l2,lq1,m1p,m2p,m1p+m2p)
                    * clebsch_gordan(l3,lq1,lq2,m3,m1+m2,m1+m2+m3)
                    * clebsch_gordan(l3,lq1,lq2,m3p,m1p+m2p,m1p+m2p+m3p)
                    * clebsch_gordan(l4,lq2,l5,m4,m1+m2+m3,-m5)
                    * clebsch_gordan(l4,lq2,l5,m4p,m1p+m2p+m3p,-m5p);
            }
        }
        num *= pow(-1, abs(m5-m5p))/(2*l5+1);

        core(map_indices[index], map_indices[index_p]) = num;
        if (index != index_p){
            core(map_indices[index_p], map_indices[index]) = num;
        }
    }
}

void Projector::order6_pre(const vector1i& l_list, std::map<int, int>& map_indices){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];

    std::set<int> nonzero_indices;
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m3p=-l3; m3p<=l3; ++m3p)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m4p=-l4; m4p<=l4; ++m4p)
    for (int m5=-l5; m5<=l5; ++m5)
    for (int m5p=-l5; m5p<=l5; ++m5p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2, m3, m4, m5};
        vector1i mv2 = {m1p, m2p, m3p, m4p, m5p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        nonzero_indices.insert(index);
        nonzero_indices.insert(index_p);
    }
    precalc_common(nonzero_indices, map_indices);
}


void Projector::order6(const vector1i& l_list){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];

    std::map<int, int> map_indices;
    order6_pre(l_list, map_indices);

    const int core_size = map_indices.size();
    core = Eigen::MatrixXd::Zero(core_size, core_size);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(10) schedule(dynamic)
    #endif
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m1p=-l1; m1p<=l1; ++m1p)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m2p=-l2; m2p<=l2; ++m2p)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m3p=-l3; m3p<=l3; ++m3p)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m4p=-l4; m4p<=l4; ++m4p)
    for (int m5=-l5; m5<=l5; ++m5)
    for (int m5p=-l5; m5p<=l5; ++m5p)
    {
        int index, index_p;
        vector1i mv1 = {m1, m2, m3, m4, m5};
        vector1i mv2 = {m1p, m2p, m3p, m4p, m5p};
        bool nonzero = check_m_nonzero(l_list, mv1, mv2, index, index_p);
        if (!nonzero)
            continue;

        int m6 = mv1[5], m6p = mv2[5];
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

        core(map_indices[index], map_indices[index_p]) = num;
        if (index != index_p){
            core(map_indices[index_p], map_indices[index]) = num;
        }
    }
}


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

double Projector::clebsch_gordan
(const int& l1, const int& l2, const int& l,
 const int& m1, const int& m2, const int& m){

    return gsl_sf_coupling_3j(2*l1, 2*l2, 2*l, 2*m1, 2*m2, -2*m)
        * sqrt(2*l+1) * pow(-1, l1-l2+m);

}

Eigen::MatrixXd& Projector::get_core(){ return core; }
const vector1i& Projector::get_row() const{ return row; }
