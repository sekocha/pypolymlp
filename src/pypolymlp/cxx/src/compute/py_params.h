/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

 ******************************************************************************/

#ifndef __PY_PARAMS
#define __PY_PARAMS

#include "mlpcpp.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void convert_params_dict_to_feature_params(const py::dict& params_dict,
                                           feature_params& fp);

#endif
