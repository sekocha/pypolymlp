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

#include "precondition.h"
#include "mlpcpp.h"

typedef std::map<tuple2, double> map_tuple2_d;
typedef std::map<tuple3, double> map_tuple3_d;
typedef std::map<tuple4, double> map_tuple4_d;
typedef std::map<tuple5, double> map_tuple5_d;
typedef std::map<tuple6, double> map_tuple6_d;
typedef std::map<tuple7, double> map_tuple7_d;
typedef std::map<tuple8, double> map_tuple8_d;


class Projector{

    Precondition pre;
    Eigen::MatrixXd core;
    int core_size;

    double clebsch_gordan(
        const int& l1, const int& l2, const int& l,
        const int& m1, const int& m2, const int& m);

    void order2(const vector1i& l_list);
    void order3(const vector1i& l_list);
    void order4(const vector1i& l_list);
    void order5(const vector1i& l_list);
    void order6(const vector1i& l_list);

    public:

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list);
    Eigen::MatrixXd& get_core();
    const vector1i& get_row() const;
};

#endif
