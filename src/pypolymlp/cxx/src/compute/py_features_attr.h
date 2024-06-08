/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PYFEATURESATTR
#define __PYFEATURESATTR

#include "mlpcpp.h"
#include "polymlp/polymlp_model_params.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyFeaturesAttr {

    vector1i radial_ids, gtinv_ids;
    vector2i tcomb_ids, polynomial_ids;
    vector3i type_comb_pair;

    public:

    PyFeaturesAttr(const py::dict& params_dict);
    ~PyFeaturesAttr();

    const vector1i& get_radial_ids() const;
    const vector1i& get_gtinv_ids() const;
    const vector2i& get_tcomb_ids() const;
    const vector2i& get_polynomial_ids() const;
    const vector3i& get_type_comb_pair() const;

};

#endif
