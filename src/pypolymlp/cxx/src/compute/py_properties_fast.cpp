/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_properties_fast.h"

PyPropertiesFast::PyPropertiesFast(const py::dict& params_dict,
                                   const vector1d& coeffs){

    convert_params_dict_to_feature_params(params_dict, fp);
    polymlp = PolymlpEval(fp, coeffs);
    polymlp_openmp = PolymlpEvalOpenMP(polymlp);
}

PyPropertiesFast::~PyPropertiesFast(){}

void PyPropertiesFast::eval(const vector2d& axis,
                            const vector2d& positions_c,
                            const vector1i& types,
                            const bool use_openmp_atom){
    /* positions_c: (3, n_atom) */
    if (use_openmp_atom == true){
        NeighborHalfOpenMP neigh(axis, positions_c, types, fp.cutoff);
        polymlp_openmp.eval(
            types, neigh.get_half_list(), neigh.get_diff_list(),
            energy, force, stress
        );
    }
    else {
        NeighborHalf neigh(axis, positions_c, types, fp.cutoff);
        polymlp.eval(
            types, neigh.get_half_list(), neigh.get_diff_list(),
            energy, force, stress
        );
    }
}

void PyPropertiesFast::eval_multiple(const vector3d& axis_array,
                                     const vector3d& positions_c_array,
                                     const vector2i& types_array){
    const int n_st = axis_array.size();
    e_array = vector1d(n_st);
    f_array = vector3d(n_st);
    s_array = vector2d(n_st);

    if (n_st > 1) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided)
        #endif
        for (int i = 0; i < n_st; ++i){
            NeighborHalf neigh(
                axis_array[i], positions_c_array[i], types_array[i], fp.cutoff
            );
            polymlp.eval(
                types_array[i], neigh.get_half_list(), neigh.get_diff_list(),
                e_array[i], f_array[i], s_array[i]
            );
        }
    }
    else if (n_st == 1) {
        NeighborHalfOpenMP neigh(
            axis_array[0], positions_c_array[0], types_array[0], fp.cutoff
        );
        polymlp_openmp.eval(
            types_array[0], neigh.get_half_list(), neigh.get_diff_list(),
            e_array[0], f_array[0], s_array[0]
        );
    }
}

/* force: (n_atom, 3) */
const double& PyPropertiesFast::get_e() const { return energy; }
const vector2d& PyPropertiesFast::get_f() const { return force; }
const vector1d& PyPropertiesFast::get_s() const { return stress; }

const vector1d& PyPropertiesFast::get_e_array() const { return e_array; }
const vector3d& PyPropertiesFast::get_f_array() const { return f_array; }
const vector2d& PyPropertiesFast::get_s_array() const { return s_array; }
