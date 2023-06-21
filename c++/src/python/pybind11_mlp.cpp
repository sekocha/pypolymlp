/***************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to
        the Free Software Foundation, Inc., 51 Franklin Street,
        Fifth Floor, Boston, MA 02110-1301, USA, or see
        http://www.gnu.org/copyleft/gpl.txt

*****************************************************************************/

#include "pybind11_mlp.h"

PYBIND11_MODULE(mlpcpp, m) {

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

    py::class_<PyModelSingleStruct>(m, "PotentialModelSingleStruct")
        .def(py::init<const py::dict&,
                      const vector2d&,
                      const vector2d&,
                      const vector1i&>())
        .def("get_x", &PyModelSingleStruct::get_x, 
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
        .def("get_type_comb_pair", &PyFeaturesAttr::get_type_comb_pair, 
                py::return_value_policy::reference_internal)
        ;

    py::class_<Readgtinv>(m, "Readgtinv")
        .def(py::init<const int&, 
                      const vector1i&, 
                      const std::vector<bool>&, 
                      const int&>())
        .def("get_lm_seq", &Readgtinv::get_lm_seq, 
                py::return_value_policy::reference_internal)
        .def("get_l_comb", &Readgtinv::get_l_comb, 
                py::return_value_policy::reference_internal)
        .def("get_lm_coeffs", &Readgtinv::get_lm_coeffs, 
                py::return_value_policy::reference_internal)
        ;

}


