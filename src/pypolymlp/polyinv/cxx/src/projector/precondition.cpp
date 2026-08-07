/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

	    Program for preparing properties required for building projector

        # Quantum theory of angular momentum (Varshalovich, p.96)

*****************************************************************************/

#include "precondition.h"


bool check_sum(const vector1i& m, const int lmax, int& mf){
    mf = - std::accumulate(m.begin(), m.end(), 0);
    if (abs(mf) > lmax)
        return false;
    return true;
}


Precondition::Precondition(){}
Precondition::Precondition(const vector1i& l_list_i)
    :l_list(l_list_i)
{

    const int order = l_list.size();
    if (order == 2) order2();
    else if (order == 3) order3();
    else if (order == 4) order4();
    else if (order == 5) order5();
    else if (order == 6) order6();
    else if (order == 7) order7();
    else if (order == 8) order8();
    else if (order == 9) order9();
    else if (order == 10) order10();

}
Precondition::~Precondition(){}

void Precondition::order2(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];

    row.clear();
    map_m_to_index2.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1){
        vector1i mv1 = {m1};
        int m2;
        if (check_sum(mv1, l2, m2)){
            mv1.emplace_back(m2);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index2[m1] = seq;
            ++seq;
        }
    }
}


void Precondition::order3(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];

    row.clear();
    map_m_to_index3.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2){
        vector1i mv1 = {m1, m2};
        int m3;
        if (check_sum(mv1, l3, m3)){
            mv1.emplace_back(m3);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index3[{m1, m2}] = seq;
            ++seq;
        }
    }
}


void Precondition::order4(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];

    row.clear();
    map_m_to_index4.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3){
        vector1i mv1 = {m1, m2, m3};
        int m4;
        if (check_sum(mv1, l4, m4)){
            mv1.emplace_back(m4);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index4[{m1, m2, m3}] = seq;
            ++seq;
        }
    }
}

void Precondition::order5(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];

    row.clear();
    map_m_to_index5.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m4=-l4; m4<=l4; ++m4){
        vector1i mv1 = {m1, m2, m3, m4};
        int m5;
        if (check_sum(mv1, l5, m5)){
            mv1.emplace_back(m5);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index5[{m1, m2, m3, m4}] = seq;
            ++seq;
        }
    }
}

void Precondition::order6(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];

    row.clear();
    map_m_to_index6.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m5=-l5; m5<=l5; ++m5){
        vector1i mv1 = {m1, m2, m3, m4, m5};
        int m6;
        if (check_sum(mv1, l6, m6)){
            mv1.emplace_back(m6);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index6[{m1, m2, m3, m4, m5}] = seq;
            ++seq;
        }
    }
}

void Precondition::order7(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];
    const int l7 = l_list[6];

    row.clear();
    map_m_to_index7.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m5=-l5; m5<=l5; ++m5)
    for (int m6=-l6; m6<=l6; ++m6){
        vector1i mv1 = {m1, m2, m3, m4, m5, m6};
        int m7;
        if (check_sum(mv1, l7, m7)){
            mv1.emplace_back(m7);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index7[{m1, m2, m3, m4, m5, m6}] = seq;
            ++seq;
        }
    }
}

void Precondition::order8(){

    const int l1 = l_list[0];
    const int l2 = l_list[1];
    const int l3 = l_list[2];
    const int l4 = l_list[3];
    const int l5 = l_list[4];
    const int l6 = l_list[5];
    const int l7 = l_list[6];
    const int l8 = l_list[7];

    row.clear();
    map_m_to_index8.clear();
    int seq(0);
    for (int m1=-l1; m1<=l1; ++m1)
    for (int m2=-l2; m2<=l2; ++m2)
    for (int m3=-l3; m3<=l3; ++m3)
    for (int m4=-l4; m4<=l4; ++m4)
    for (int m5=-l5; m5<=l5; ++m5)
    for (int m6=-l6; m6<=l6; ++m6)
    for (int m7=-l7; m7<=l7; ++m7){
        vector1i mv1 = {m1, m2, m3, m4, m5, m6, m7};
        int m8;
        if (check_sum(mv1, l8, m8)){
            mv1.emplace_back(m8);
            int index = lm_to_matrix_index(mv1);
            row.emplace_back(index);
            map_m_to_index8[{m1, m2, m3, m4, m5, m6, m7}] = seq;
            ++seq;
        }
    }
}

int Precondition::lm_to_matrix_index(const vector1i& m_array) {
    /***
    The original code is as follows.

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
    ***/

    int index = 0;
    long long multiplier = 1;
    int size = l_list.size();

    for (int i = size - 1; i >= 0; --i) {
        index += (m_array[i] + l_list[i]) * multiplier;
        multiplier *= (2 * l_list[i] + 1);
    }
    return index;
}

void Precondition::order9(){}
void Precondition::order10(){}

const vector1i& Precondition::get_row() const{ return row; }


std::map<int, int>& Precondition::get_map_m_to_index2(){
    return map_m_to_index2;
};
map_tuple2_i& Precondition::get_map_m_to_index3() {
    return map_m_to_index3;
};
map_tuple3_i& Precondition::get_map_m_to_index4() {
    return map_m_to_index4;
};
map_tuple4_i& Precondition::get_map_m_to_index5() {
    return map_m_to_index5;
};
map_tuple5_i& Precondition::get_map_m_to_index6() {
    return map_m_to_index6;
};
map_tuple6_i& Precondition::get_map_m_to_index7() {
    return map_m_to_index7;
};
map_tuple7_i& Precondition::get_map_m_to_index8() {
    return map_m_to_index8;
};
map_tuple8_i& Precondition::get_map_m_to_index9() {
    return map_m_to_index9;
};
map_tuple9_i& Precondition::get_map_m_to_index10() {
    return map_m_to_index10;
};
