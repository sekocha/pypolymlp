/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "py_properties_fast.h"

PyPropertiesFast::PyPropertiesFast(
    const py::dict& params_dict, const vector1d& coeffs
){
    convert_params_dict_to_feature_params(params_dict, fp);
    polymlp = PolymlpEval(fp, coeffs);
}

PyPropertiesFast::~PyPropertiesFast(){}

void PyPropertiesFast::eval(
    const vector2d& axis,
    const vector2d& positions_c,
    const vector1i& types,
    const bool use_openmp_atom
){
    /* positions_c: (3, n_atom) */
    NeighborHalf neigh(axis, positions_c, fp.cutoff, use_openmp_atom);
    polymlp.eval(types, neigh, use_openmp_atom, energy, force, stress);
}

void PyPropertiesFast::eval_multiple(
    const vector3d& axis_array,
    const vector3d& positions_c_array,
    const vector2i& types_array
){
    const int n_st = axis_array.size();
    e_array = vector1d(n_st);
    f_array = vector3d(n_st);
    s_array = vector2d(n_st);
    bool use_openmp_structure = false;
    if (!use_openmp_structure){
        const bool use_openmp = true;
        for (int i = 0; i < n_st; ++i){
            NeighborHalf neigh(
                axis_array[i],
                positions_c_array[i],
                fp.cutoff,
                use_openmp
            );
            polymlp.eval(
                types_array[i],
                neigh,
                use_openmp,
                e_array[i],
                f_array[i],
                s_array[i]);
        }
    }
    else {
        // Deprecated: OPENMP parallelization sometimes fails in pytest autocalc
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided)
        #endif
        for (int i = 0; i < n_st; ++i){
            const bool use_openmp = false;
            NeighborHalf neigh(
                axis_array[i],
                positions_c_array[i],
                fp.cutoff,
                use_openmp
            );
            polymlp.eval(
                types_array[i], neigh, use_openmp,
                e_array[i], f_array[i], s_array[i]);
        }
    }
}

/* force: (n_atom, 3) */
const double& PyPropertiesFast::get_e() const { return energy; }
const vector2d& PyPropertiesFast::get_f() const { return force; }
const vector1d& PyPropertiesFast::get_s() const { return stress; }

const vector1d& PyPropertiesFast::get_e_array() const { return e_array; }
const vector3d& PyPropertiesFast::get_f_array() const { return f_array; }
const vector2d& PyPropertiesFast::get_s_array() const { return s_array; }
