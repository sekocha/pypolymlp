/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PYPROPERTIES
#define __PYPROPERTIES

#include "mlpcpp.h"
#include "compute/neighbor.h"
#include "compute/model_properties.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyProperties {

    Eigen::VectorXd e_all;
    vector2d f_all;
    Eigen::MatrixXd s_all;

    public:

    PyProperties(const py::dict& params_dict,
                 const vector1d& coeffs,
                 const vector3d& axis,
                 const vector3d& positions_c,
                 const vector2i& types);

    ~PyProperties();

    Eigen::VectorXd& get_e();
    const vector2d& get_f() const;
    Eigen::MatrixXd& get_s();

};

#endif
