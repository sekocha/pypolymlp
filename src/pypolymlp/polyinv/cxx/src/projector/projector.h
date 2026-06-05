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

    vector2i m_list;
    vector1i index_list;

    std::map<std::tuple<int, int, int, int>, int> map_m_to_index;


    int lm_to_matrix_index(const vector1i& l_list, const vector1i& m_array);
    bool check_m_nonzero(
        const vector1i& l_list,
        vector1i& mv1,
        vector1i& mv2,
        int& index,
        int& index_p);

    void order2(const vector1i& l_list);
    void order3(const vector1i& l_list);
    void order4(const vector1i& l_list);
    void order5(const vector1i& l_list);
    void order6(const vector1i& l_list);

    void precalc_common(
        const std::set<int>& nonzero_indices,
        std::map<int, int>& map_indices);

    void order2_pre(const vector1i& l_list, std::map<int, int>& map_indices);
    void order3_pre(const vector1i& l_list, std::map<int, int>& map_indices);
    void order4_pre(const vector1i& l_list, std::map<int, int>& map_indices);
    void order5_pre(const vector1i& l_list, std::map<int, int>& map_indices);
    void order6_pre(const vector1i& l_list, std::map<int, int>& map_indices);

    double clebsch_gordan
        (const int& l1, const int& l2, const int& l,
         const int& m1, const int& m2, const int& m);

    public:

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list);
    Eigen::MatrixXd& get_core();
    const vector1i& get_row() const;
};

#endif
