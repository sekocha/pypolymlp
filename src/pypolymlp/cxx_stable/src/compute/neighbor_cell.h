/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __NEIGHBOR_CELL
#define __NEIGHBOR_CELL

#include "mlpcpp.h"


class NeighborCell{

    vector2d trans_c_array;
    vector2d axis, axis_inv, positions_c;
    double cutoff;

    vector1d axis0, axis1, axis2;
    double dot00, dot11, dot22, dot01, dot02, dot12;
    bool ref01, ref02, ref12;
    int ref_sum;

    int compute_metric();
    vector1d get_axis_vector(const int col);
    int replace_axis_vector(const vector1i& vec, const int col);

    vector1d dot_prod(const vector2d& mat, const vector1d& vec2);

    double optimize_1d_axis0(const int val1, const int val2, vector1i& vec);
    double optimize_1d_axis1(const int val1, const int val2, vector1i& vec);
    double optimize_1d_axis2(const int val1, const int val2, vector1i& vec);

    int calc_inverse_axis();
    int standardize_positions(vector2d& positions);
    int refine_axis();
    double find_maximum_diagonal_in_cell();

    int find_trans();

    template <typename T> T dot_prod(const std::vector<T>& vec1,
                                     const std::vector<T>& vec2);
    template <typename T> T norm(const std::vector<T>& vec);
    template <typename T> double distance(const T& i, const T& j, const T& k);
    template <typename T> vector1d to_cartesian(const T& i,
                                                const T& j,
                                                const T& k);

    public:

    NeighborCell(const vector2d& axis_i,
                 const vector2d& positions_c_i,
                 const double cutoff_i);
    ~NeighborCell();

    const vector2d& get_translations() const;
    const vector2d& get_axis() const;
    const vector2d& get_positions_cartesian() const;

};

template <typename T>
T NeighborCell::dot_prod(const std::vector<T>& vec1,
                         const std::vector<T>& vec2){
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}


template <typename T>
T NeighborCell::norm(const std::vector<T>& vec){
    return sqrt(dot_prod(vec, vec));
}

template <typename T>
double NeighborCell::distance(const T& i, const T& j, const T& k){

    double val1 = axis[0][0] * i + axis[0][1] * j + axis[0][2] * k;
    double val2 = axis[1][0] * i + axis[1][1] * j + axis[1][2] * k;
    double val3 = axis[2][0] * i + axis[2][1] * j + axis[2][2] * k;
    return sqrt(val1 * val1 + val2 * val2 + val3 * val3);
}

template <typename T>
vector1d NeighborCell::to_cartesian(const T& i, const T& j, const T& k){

    vector1d vec(3);
    vec[0] = axis[0][0] * i + axis[0][1] * j + axis[0][2] * k;
    vec[1] = axis[1][0] * i + axis[1][1] * j + axis[1][2] * k;
    vec[2] = axis[2][0] * i + axis[2][1] * j + axis[2][2] * k;
    return vec;
}


#endif
