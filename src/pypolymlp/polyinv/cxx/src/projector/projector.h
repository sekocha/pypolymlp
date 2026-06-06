/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PROJECTOR
#define __PROJECTOR

#include <set>
#include <map>
#include <tuple>
#include <mutex>
#include <omp.h>

#include <gsl/gsl_sf_coupling.h>
#include <Eigen/Dense>

#include "mlpcpp.h"


class Projector{

    Eigen::MatrixXd core;
    vector1i row;

    std::map<int, int> map_m_to_index2;
    std::map<std::tuple<int, int>, int> map_m_to_index3;
    std::map<std::tuple<int, int, int>, int> map_m_to_index4;
    std::map<std::tuple<int, int, int, int>, int> map_m_to_index5;
    std::map<std::tuple<int, int, int, int, int>, int> map_m_to_index6;

    int lm_to_matrix_index(const vector1i& l_list, const vector1i& m_array);
    bool check_sum(const vector1i& l_list, const vector1i& m, int& mf);

    double clebsch_gordan(
        const int& l1, const int& l2, const int& l,
        const int& m1, const int& m2, const int& m);

    void order2(const vector1i& l_list);
    void order3(const vector1i& l_list);
    void order4(const vector1i& l_list);
    void order5(const vector1i& l_list);
    void order6(const vector1i& l_list);

    void order2_pre(const vector1i& l_list);
    void order3_pre(const vector1i& l_list);
    void order4_pre(const vector1i& l_list);
    void order5_pre(const vector1i& l_list);
    void order6_pre(const vector1i& l_list);

    public:

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list);
    Eigen::MatrixXd& get_core();
    const vector1i& get_row() const;
};

#endif
