/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PROJECTOR
#define __PROJECTOR

#include <gsl/gsl_sf_coupling.h>

#include "mlpcpp.h"

class Projector{

    vector1i row, col;
    vector1d data;

    int lm_to_matrix_index(const vector1i& l_list, const vector1i& m_array);

    vector2i mlist_nonzero(const vector1i& l_list, const vector2i& m_list);

    void order2(const vector1i& l_list, const vector2i& m_list);
    void order3(const vector1i& l_list, const vector2i& m_list);
    void order4(const vector1i& l_list, const vector2i& m_list);
    void order5(const vector1i& l_list, const vector2i& m_list);
    void order6(const vector1i& l_list, const vector2i& m_list);

    void array_initialize(const vector2i& m_list);

    public:

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list, const vector2i& m_list);

    double clebsch_gordan
        (const int& l1, const int& l2, const int& l,
         const int& m1, const int& m2, const int& m);

    const vector1i& get_row() const;
    const vector1i& get_col() const;
    const vector1d& get_data() const;
};

#endif
