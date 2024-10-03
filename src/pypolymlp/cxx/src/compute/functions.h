/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

 ******************************************************************************/

#ifndef __FUNCTIONS
#define __FUNCTIONS

#include "mlpcpp.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void compute_products(
    const vector2i& map, const vector1dc& element, vector1dc& prod_vals
);

void compute_products_real(
    const vector2i& map, const vector1dc& element, vector1d& prod_vals
);

double prod_real(const dc& val1, const dc& val2);

void convert_params_dict_to_feature_params(const py::dict& params_dict,
                                           feature_params& fp);

#endif
