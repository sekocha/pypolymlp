/****************************************************************************

        Copyright (C) 2017 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

	    Main program for building projector for group-theoretic invariants

        # Quantum theory of angular momentum (Varshalovich, p.96)

*****************************************************************************/

#include "projector.h"

Projector::Projector(){}
Projector::~Projector(){}

void Projector::build_projector
(const vector1i& l_list, const vector2i& m_list){

    vector2i m_list_nonzero = mlist_nonzero(l_list, m_list);
    int order = l_list.size();
    if (order == 2)
        order2(l_list, m_list_nonzero);
    else if (order == 3)
        order3(l_list, m_list_nonzero);
    else if (order == 4)
        order4(l_list, m_list_nonzero);
    else if (order == 5)
        order5(l_list, m_list_nonzero);
    else if (order == 6)
        order6(l_list, m_list_nonzero);

}

vector2i Projector::mlist_nonzero
(const vector1i& l_list, const vector2i& m_list){

    vector2i output;
    for (int i = 0; i < m_list.size(); ++i){
        int index, index_p, mf, mfp;
        vector1i mv1, mv2;
        for (int j = 0; j < m_list[i].size(); ++j){
            if (j%2==0)
                mv1.push_back(m_list[i][j]);
            else
                mv2.push_back(m_list[i][j]);
        }
        mf = - std::accumulate(mv1.begin(), mv1.end(), 0);
        mfp = - std::accumulate(mv2.begin(), mv2.end(), 0);
        mv1.push_back(int(mf));
        mv2.push_back(int(mfp));
        index = lm_to_matrix_index(l_list, mv1);
        index_p = lm_to_matrix_index(l_list, mv2);
        if (index <= index_p and abs(mf) <= l_list[l_list.size()-1]
            and abs(mfp) <= l_list[l_list.size()-1]){
            std::copy(mv2.begin(),mv2.end(),std::back_inserter(mv1));
            output.push_back(mv1);
        }
    }
    return output;

}

void Projector::array_initialize(const vector2i& m_list){

    int size = m_list.size()*2;
    row.clear();
    col.clear();
    data.clear();
    row.resize(size);
    col.resize(size);
    data.resize(size);

}

void Projector::order2
(const vector1i& l_list, const vector2i& m_list){

    array_initialize(m_list);

    int l1, l2;
    l1 = l_list[0]; l2 = l_list[1];
    for (int i = 0; i < m_list.size(); ++i){
        int m1, m2, m1p, m2p, index, index_p;
        m1 = m_list[i][0]; m2 = m_list[i][1];
        m1p = m_list[i][2]; m2p = m_list[i][3];
        vector1i mv1, mv2;
        for (int j = 0; j < m_list[i].size()/2; ++j)
            mv1.push_back(m_list[i][j]);
        for (int j = m_list[i].size()/2; j < m_list[i].size(); ++j)
            mv2.push_back(m_list[i][j]);

        double num;
        if (l1 == l2 and -m1 == m2 and -m1p == m2p){
            num = pow(-1, abs(m2 - m2p)) / (2 * l2 + 1);
        }
        else num = 0.0;

        index = lm_to_matrix_index(l_list, mv1);
        index_p = lm_to_matrix_index(l_list, mv2);

        row[i*2] = index;
        col[i*2] = index_p;
        data[i*2] = num;
        row[i*2+1] = index_p;
        col[i*2+1] = index;
        if (index != index_p)
            data[i*2+1] = num;
        else
            data[i*2+1] = 0.0;
    }
}

void Projector::order3
(const vector1i& l_list, const vector2i& m_list){

    array_initialize(m_list);

    int l1, l2, l3;
    l1 = l_list[0]; l2 = l_list[1]; l3 = l_list[2];
    #ifdef _OPENMP
    #pragma omp parallel for shared(l1, l2, l3)
    #endif
    for (int i = 0; i < m_list.size(); ++i){
        int m1, m2, m3, m1p, m2p, m3p, index, index_p;
        m1 = m_list[i][0]; m2 = m_list[i][1]; m3 = m_list[i][2];
        m1p = m_list[i][3]; m2p = m_list[i][4]; m3p = m_list[i][5];
        vector1i mv1, mv2;
        for (int j = 0; j < m_list[i].size()/2; ++j)
            mv1.push_back(m_list[i][j]);
        for (int j = m_list[i].size()/2; j < m_list[i].size(); ++j)
            mv2.push_back(m_list[i][j]);

        double num = pow(-1, abs(m3-m3p))/(2*l3+1)
              * clebsch_gordan(l1, l2, l3, m1, m2, -m3)
              * clebsch_gordan(l1, l2, l3, m1p, m2p, -m3p);

        index = lm_to_matrix_index(l_list, mv1);
        index_p = lm_to_matrix_index(l_list, mv2);

        row[i*2] = index;
        col[i*2] = index_p;
        data[i*2] = num;
        row[i*2+1] = index_p;
        col[i*2+1] = index;
        if (index != index_p)
            data[i*2+1] = num;
        else
            data[i*2+1] = 0.0;
    }
}

void Projector::order4
(const vector1i& l_list, const vector2i& m_list){

    array_initialize(m_list);

    int l1, l2, l3, l4;
    l1 = l_list[0]; l2 = l_list[1]; l3 = l_list[2]; l4 = l_list[3];
    #ifdef _OPENMP
    #pragma omp parallel for shared(l1, l2, l3, l4)
    #endif
    for (int i = 0; i < m_list.size(); ++i){
        int m1, m2, m3, m4, m1p, m2p, m3p, m4p, index, index_p;
        m1 = m_list[i][0]; m2 = m_list[i][1];
        m3 = m_list[i][2]; m4 = m_list[i][3];
        m1p = m_list[i][4]; m2p = m_list[i][5];
        m3p = m_list[i][6]; m4p = m_list[i][7];
        vector1i mv1, mv2;
        for (int j = 0; j < m_list[i].size()/2; ++j)
            mv1.push_back(m_list[i][j]);
        for (int j = m_list[i].size()/2; j < m_list[i].size(); ++j)
            mv2.push_back(m_list[i][j]);

        double num(0);
        for (int l = abs(l1-l2); l < l1+l2+1; ++l){
            num += clebsch_gordan(l1, l2, l, m1, m2, -m3-m4)
                * clebsch_gordan(l1, l2, l, m1p, m2p, -m3p-m4p)
                * clebsch_gordan(l3, l, l4, m3, -m3-m4, -m4)
                * clebsch_gordan(l3, l, l4, m3p, -m3p-m4p, -m4p);
        }
        num *= pow(-1, abs(m4-m4p))/(2*l4+1);

        index = lm_to_matrix_index(l_list, mv1);
        index_p = lm_to_matrix_index(l_list, mv2);

        row[i*2] = index;
        col[i*2] = index_p;
        data[i*2] = num;
        row[i*2+1] = index_p;
        col[i*2+1] = index;
        if (index != index_p)
            data[i*2+1] = num;
        else
            data[i*2+1] = 0.0;
    }
}

void Projector::order5
(const vector1i& l_list, const vector2i& m_list){

    array_initialize(m_list);

    int l1, l2, l3, l4, l5;
    l1 = l_list[0]; l2 = l_list[1]; l3 = l_list[2];
    l4 = l_list[3]; l5 = l_list[4];
    #ifdef _OPENMP
    #pragma omp parallel for shared(l1, l2, l3, l4, l5)
    #endif
    for (int i = 0; i < m_list.size(); ++i){
        int m1, m2, m3, m4, m5, m1p, m2p, m3p, m4p, m5p, index, index_p;
        m1 = m_list[i][0]; m2 = m_list[i][1];
        m3 = m_list[i][2]; m4 = m_list[i][3]; m5 = m_list[i][4];
        m1p = m_list[i][5]; m2p = m_list[i][6];
        m3p = m_list[i][7]; m4p = m_list[i][8]; m5p = m_list[i][9];
        vector1i mv1, mv2;
        for (int j = 0; j < m_list[i].size()/2; ++j)
            mv1.push_back(m_list[i][j]);
        for (int j = m_list[i].size()/2; j < m_list[i].size(); ++j)
            mv2.push_back(m_list[i][j]);

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

        index = lm_to_matrix_index(l_list, mv1);
        index_p = lm_to_matrix_index(l_list, mv2);

        row[i*2] = index;
        col[i*2] = index_p;
        data[i*2] = num;
        row[i*2+1] = index_p;
        col[i*2+1] = index;
        if (index != index_p)
            data[i*2+1] = num;
        else
            data[i*2+1] = 0.0;
    }
}

void Projector::order6
(const vector1i& l_list, const vector2i& m_list){

    array_initialize(m_list);

    int l1, l2, l3, l4, l5, l6;
    l1 = l_list[0]; l2 = l_list[1]; l3 = l_list[2];
    l4 = l_list[3]; l5 = l_list[4]; l6 = l_list[5];
    #ifdef _OPENMP
    #pragma omp parallel for shared(l1, l2, l3, l4, l5, l6)
    #endif
    for (int i = 0; i < m_list.size(); ++i){
        int m1, m2, m3, m4, m5, m6, m1p, m2p, m3p, m4p, m5p, m6p,
            index, index_p;
        m1 = m_list[i][0]; m2 = m_list[i][1]; m3 = m_list[i][2];
        m4 = m_list[i][3]; m5 = m_list[i][4]; m6 = m_list[i][5];
        m1p = m_list[i][6]; m2p = m_list[i][7]; m3p = m_list[i][8];
        m4p = m_list[i][9]; m5p = m_list[i][10]; m6p = m_list[i][11];
        vector1i mv1, mv2;
        for (int j = 0; j < m_list[i].size()/2; ++j)
            mv1.push_back(m_list[i][j]);
        for (int j = m_list[i].size()/2; j < m_list[i].size(); ++j)
            mv2.push_back(m_list[i][j]);

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

        index = lm_to_matrix_index(l_list, mv1);
        index_p = lm_to_matrix_index(l_list, mv2);

        row[i*2] = index;
        col[i*2] = index_p;
        data[i*2] = num;
        row[i*2+1] = index_p;
        col[i*2+1] = index;
        if (index != index_p)
            data[i*2+1] = num;
        else
            data[i*2+1] = 0.0;
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

const vector1i& Projector::get_row() const{ return row; }
const vector1i& Projector::get_col() const{ return col; }
const vector1d& Projector::get_data() const{ return data; }
