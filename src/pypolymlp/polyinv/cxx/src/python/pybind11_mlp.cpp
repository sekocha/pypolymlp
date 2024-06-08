/***************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "pybind11_mlp.h"

PYBIND11_MODULE(libprojcpp, m) {

    py::class_<Projector>(m, "Projector")
        .def(py::init<>())
        .def("build_projector", &Projector::build_projector)
        .def("get_row", &Projector::get_row,
            py::return_value_policy::reference_internal)
        .def("get_col", &Projector::get_col,
            py::return_value_policy::reference_internal)
        .def("get_data", &Projector::get_data,
            py::return_value_policy::reference_internal)
        ;

}
