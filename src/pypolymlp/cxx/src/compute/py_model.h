/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PYMODEL
#define __PYMODEL

#include "mlpcpp.h"
#include "polymlp/polymlp_model_params.h"
#include "polymlp/polymlp_features.h"
#include "compute/py_params.h"
#include "compute/neighbor.h"
#include "compute/model_fast.h"
#include "compute/features.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyModel {

    Eigen::MatrixXd x_all;
    vector1i xf_begin_dataset, xs_begin_dataset;
    vector1i n_data;

    void set_index(const std::vector<int>& n_data_dataset,
                   const std::vector<bool>& force_dataset,
                   const std::vector<int>& n_atoms_st,
                   std::vector<int>& xf_begin,
                   std::vector<int>& xs_begin,
                   std::vector<bool>& force);

    public:

    PyModel(const py::dict& params_dict,
            const vector3d& axis,
            const vector3d& positions_c,
            const vector2i& types,
            const vector1i& n_st_dataset,
            const std::vector<bool>& force_dataset,
            const vector1i& n_atoms_all);

    ~PyModel();

    Eigen::MatrixXd& get_x();
    const vector1i& get_fbegin() const;
    const vector1i& get_sbegin() const;
    const vector1i& get_n_data() const;

};

#endif
