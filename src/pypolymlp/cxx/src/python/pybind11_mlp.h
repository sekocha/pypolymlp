/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __PYBIND11_MLP
#define __PYBIND11_MLP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "mlpcpp.h"
#include "compute/py_model.h"
#include "compute/py_hybrid_model.h"
#include "compute/py_properties_fast.h"
#include "compute/py_features_attr.h"
#include "compute/neighbor.h"
#include "polymlp/polymlp_read_gtinv.h"

// For tests
#include "compute/neighbor_half.h"
#include "compute/neighbor_half_openmp.h"
#include "polymlp/polymlp_mlipkk_spherical_harmonics.h"
#include "polymlp/polymlp_mlpcpp.h"

namespace py = pybind11;

#endif
