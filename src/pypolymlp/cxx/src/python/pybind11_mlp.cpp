/***************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "pybind11_mlp.h"

PYBIND11_MODULE(libmlpcpp, m) {

    py::class_<PyModel>(m, "PotentialModel")
        .def(py::init<const py::dict&,
                      const vector3d&,
                      const vector3d&,
                      const vector2i&,
                      const vector1i&,
                      const std::vector<bool>&,
                      const vector1i&>())
        .def("get_x", &PyModel::get_x,
                py::return_value_policy::reference_internal)
        .def("get_fbegin", &PyModel::get_fbegin,
                py::return_value_policy::reference_internal)
        .def("get_sbegin", &PyModel::get_sbegin,
                py::return_value_policy::reference_internal)
        .def("get_n_data", &PyModel::get_n_data,
                py::return_value_policy::reference_internal)
        ;

    py::class_<PyAdditiveModel>(m, "PotentialAdditiveModel")
        .def(py::init<const std::vector<py::dict>&,
                      const vector3d&,
                      const vector3d&,
                      const vector2i&,
                      const vector1i&,
                      const std::vector<bool>&,
                      const vector1i&>())
        .def("get_x", &PyAdditiveModel::get_x,
                py::return_value_policy::reference_internal)
        .def("get_fbegin", &PyAdditiveModel::get_fbegin,
                py::return_value_policy::reference_internal)
        .def("get_sbegin", &PyAdditiveModel::get_sbegin,
                py::return_value_policy::reference_internal)
        .def("get_cumulative_n_features",
                &PyAdditiveModel::get_cumulative_n_features,
                py::return_value_policy::reference_internal)
        .def("get_n_data", &PyAdditiveModel::get_n_data,
                py::return_value_policy::reference_internal)
        ;

    py::class_<PyPropertiesFast>(m, "PotentialPropertiesFast")
        .def(py::init<const py::dict&,
                      const vector1d&>())
        .def("eval", &PyPropertiesFast::eval)
        .def("eval_multiple", &PyPropertiesFast::eval_multiple)
        .def("get_e", &PyPropertiesFast::get_e,
                py::return_value_policy::reference_internal)
        .def("get_f", &PyPropertiesFast::get_f,
                py::return_value_policy::reference_internal)
        .def("get_s", &PyPropertiesFast::get_s,
                py::return_value_policy::reference_internal)
        .def("get_e_array", &PyPropertiesFast::get_e_array,
                py::return_value_policy::reference_internal)
        .def("get_f_array", &PyPropertiesFast::get_f_array,
                py::return_value_policy::reference_internal)
        .def("get_s_array", &PyPropertiesFast::get_s_array,
                py::return_value_policy::reference_internal)
        ;

    py::class_<PyFeaturesAttr>(m, "FeaturesAttr")
        .def(py::init<const py::dict&>())
        .def("get_radial_ids", &PyFeaturesAttr::get_radial_ids,
                py::return_value_policy::reference_internal)
        .def("get_gtinv_ids", &PyFeaturesAttr::get_gtinv_ids,
                py::return_value_policy::reference_internal)
        .def("get_tcomb_ids", &PyFeaturesAttr::get_tcomb_ids,
                py::return_value_policy::reference_internal)
        .def("get_polynomial_ids", &PyFeaturesAttr::get_polynomial_ids,
                py::return_value_policy::reference_internal)
        .def("get_type_pairs", &PyFeaturesAttr::get_type_pairs,
                py::return_value_policy::reference_internal)
        ;

    py::class_<Readgtinv>(m, "Readgtinv")
        .def(py::init<const int&,
                      const vector1i&,
                      const std::vector<bool>&,
                      const int&,
                      const int&>())
        .def("get_lm_seq", &Readgtinv::get_lm_seq,
                py::return_value_policy::reference_internal)
        .def("get_l_comb", &Readgtinv::get_l_comb,
                py::return_value_policy::reference_internal)
        .def("get_lm_coeffs", &Readgtinv::get_lm_coeffs,
                py::return_value_policy::reference_internal)
        ;

    py::class_<Neighbor>(m, "Neighbor")
        .def(py::init<const vector2d&,
                      const vector2d&,
                      const vector1i&,
                      const int&,
                      const double&>())
        .def("get_distances", &Neighbor::get_dis_array,
                py::return_value_policy::reference_internal)
        .def("get_differences", &Neighbor::get_diff_array,
                py::return_value_policy::reference_internal)
        .def("get_neighbor_indices", &Neighbor::get_atom2_array,
                py::return_value_policy::reference_internal)
        ;

}
