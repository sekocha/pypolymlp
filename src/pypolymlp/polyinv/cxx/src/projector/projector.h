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

struct Key3 {
    int i, j, k;
    bool operator==(const Key3& other) const {
        return i == other.i
            && j == other.j
            && k == other.k;
    }
};

struct Key4 {
    int i, j, k, l;
    bool operator==(const Key4& other) const {
        return i == other.i
            && j == other.j
            && k == other.k
            && l == other.l;
    }
};

struct Hash3 {
    size_t operator()(const Key3& x) const {
        return (static_cast<size_t>(x.i & 0x3FF) << 20)
             | (static_cast<size_t>(x.j & 0x3FF) << 10)
             |  static_cast<size_t>(x.k & 0x3FF);
    }
};

struct Hash4 {
    size_t operator()(const Key4& x) const {
        return (static_cast<size_t>(x.i & 0x3FF) << 30)
             | (static_cast<size_t>(x.j & 0x3FF) << 20)
             | (static_cast<size_t>(x.k & 0x3FF) << 10)
             |  static_cast<size_t>(x.l & 0x3FF);
    }
};

typedef std::unordered_map<Key3,double,Hash3> map3;
typedef std::unordered_map<Key4,double,Hash4> map4;


class Projector {

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
    void order7(const vector1i& l_list);
    void order8(const vector1i& l_list);
    void order9(const vector1i& l_list);
    void order10(const vector1i& l_list);

    void set_inter_prod_first(
        map3& cleb,
        const int l1,
        const int l2,
        const vector1i& list_m,
        const vector1i& list_mp,
        vector1d& prod_lq,
        vector1i& list_lq);

    void set_inter_prod(
        const vector1d& prod_lq_prev,
        const vector1i& list_lq_prev,
        map4& cleb,
        const int l,
        const vector1i& list_m,
        const vector1i& list_mp,
        vector1d& prod_lq,
        vector1i& list_lq);

    double set_final_prod(
        const vector1d& prod_lq_prev,
        const vector1i& list_lq_prev,
        map3& cleb,
        const vector1i& list_m,
        const vector1i& list_mp);

    void assign_core(
        Eigen::MatrixXd& core,
        const double num,
        const int index,
        const int index_p,
        const int index2,
        const int index_p2);

    public:

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list);
    Eigen::MatrixXd& get_core();
    const vector1i& get_row() const;
};

#endif
