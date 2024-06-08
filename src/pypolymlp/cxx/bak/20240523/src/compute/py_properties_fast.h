/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PYPROPERTIES_FAST
#define __PYPROPERTIES_FAST

#include "mlpcpp.h"
#include "compute/neighbor_half.h"
#include "compute/polymlp_eval.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyPropertiesFast {

    vector1d e_array;
    vector3d f_array;
    vector2d s_array;

    double energy;
    vector2d force;
    vector1d stress;

    struct feature_params fp;
    PolymlpEval polymlp;

    public:

    PyPropertiesFast(const py::dict& params_dict, const vector1d& coeffs);
    ~PyPropertiesFast();

    void eval(const vector2d& axis,
              const vector2d& positions_c,
              const vector1i& types);

    void eval_multiple(const vector3d& axis_array,
                       const vector3d& positions_c_array,
                       const vector2i& types_array);

    const double& get_e() const;
    const vector2d& get_f() const;
    const vector1d& get_s() const;

    const vector1d& get_e_array() const;
    const vector3d& get_f_array() const;
    const vector2d& get_s_array() const;

};

#endif
