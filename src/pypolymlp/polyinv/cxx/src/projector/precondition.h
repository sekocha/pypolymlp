/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PRECONDITION
#define __PRECONDITION

#include <map>
#include <tuple>
#include "mlpcpp.h"


typedef std::tuple<int, int> tuple2;
typedef std::tuple<int, int, int> tuple3;
typedef std::tuple<int, int, int, int> tuple4;
typedef std::tuple<int, int, int, int, int> tuple5;
typedef std::tuple<int, int, int, int, int, int> tuple6;
typedef std::tuple<int, int, int, int, int, int, int> tuple7;
typedef std::tuple<int, int, int, int, int, int, int, int> tuple8;

typedef std::map<tuple2, int> map_tuple2_i;
typedef std::map<tuple3, int> map_tuple3_i;
typedef std::map<tuple4, int> map_tuple4_i;
typedef std::map<tuple5, int> map_tuple5_i;
typedef std::map<tuple6, int> map_tuple6_i;
typedef std::map<tuple7, int> map_tuple7_i;
typedef std::map<tuple8, int> map_tuple8_i;

bool check_sum(const vector1i& m, const int lmax, int& mf);

class Precondition{

    vector1i l_list;
    vector1i row;

    std::map<int, int> map_m_to_index2;
    map_tuple2_i map_m_to_index3;
    map_tuple3_i map_m_to_index4;
    map_tuple4_i map_m_to_index5;
    map_tuple5_i map_m_to_index6;
    map_tuple6_i map_m_to_index7;
    map_tuple7_i map_m_to_index8;

    int lm_to_matrix_index(const vector1i& m_array);

    void order2();
    void order3();
    void order4();
    void order5();
    void order6();
    void order7();
    void order8();

    public:

    Precondition();
    Precondition(const vector1i& l_list_i);
    ~Precondition();

    const vector1i& get_row() const;

    std::map<int, int>& get_map_m_to_index2();
    map_tuple2_i& get_map_m_to_index3();
    map_tuple3_i& get_map_m_to_index4();
    map_tuple4_i& get_map_m_to_index5();
    map_tuple5_i& get_map_m_to_index6();
    map_tuple6_i& get_map_m_to_index7();
    map_tuple7_i& get_map_m_to_index8();
};

#endif
