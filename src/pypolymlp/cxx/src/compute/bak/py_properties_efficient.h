/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PYPROPERTIES_EFFICIENT
#define __PYPROPERTIES_EFFICIENT

#include "mlpcpp.h"
#include "compute/neighbor_half.h"
#include "compute/polymlp_eval.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyPropertiesFast {

    Eigen::VectorXd e_all;
    vector3d f_all;
    Eigen::MatrixXd s_all;

    public: 

    PyPropertiesFast(const py::dict& params_dict,
                     const vector1d& coeffs, 
                     const vector3d& axis, 
                     const vector3d& positions_c,
                     const vector2i& types);

    ~PyPropertiesFast();

    Eigen::VectorXd& get_e();
    const vector3d& get_f() const;
    Eigen::MatrixXd& get_s();

};

#endif
