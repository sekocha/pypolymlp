/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PROJECTOR
#define __PROJECTOR

#include <gsl/gsl_sf_coupling.h>
#include <omp.h>

#include "mlpcpp.h"


class Projector{

    vector1i row, col;
    vector1d data;

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

    void add(
        const int index,
        const int index_p,
        const double num,
        vector1i& row_vec,
        vector1i& col_vec,
        vector1d& data_vec);

    void collect(vector2i& rows, vector2i& cols, vector2d& vals);
    void array_initialize();

    double clebsch_gordan
        (const int& l1, const int& l2, const int& l,
         const int& m1, const int& m2, const int& m);

    public:

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list);
    const vector1i& get_row() const;
    const vector1i& get_col() const;
    const vector1d& get_data() const;
};

#endif
