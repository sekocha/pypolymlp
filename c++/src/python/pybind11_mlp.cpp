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

    py::class_<ModelPy>(m, "PotentialModel")
        .def(py::init<const vector3d&,
                      const vector3d&,
                      const vector2i&,
                      const int&, 
                      const bool&,
                      const vector2d&,
                      const double&,
                      const std::string&,
                      const std::string&,
                      const int&,
                      const int&,
                      const int&,
                      const vector3i&,
                      const vector2i&,
                      const vector2d&,
                      const vector1i&,
                      const std::vector<bool>&,
                      const vector1i&,
                      const bool&,
                      const bool&>())
        .def("get_x", &ModelPy::get_x, 
                py::return_value_policy::reference_internal)
        .def("get_fbegin", &ModelPy::get_fbegin, 
                py::return_value_policy::reference_internal)
        .def("get_sbegin", &ModelPy::get_sbegin, 
                py::return_value_policy::reference_internal)
        ;

    py::class_<ModelPCAPy>(m, "PotentialModelPCA")
        .def(py::init<const vector3d&,
                      const vector3d&,
                      const vector2i&,
                      const int&, 
                      const bool&,
                      const vector2d&,
                      const double&,
                      const std::string&,
                      const std::string&,
                      const int&,
                      const int&,
                      const int&,
                      const vector3i&,
                      const vector2i&,
                      const vector2d&,
                      const vector1i&,
                      const std::vector<bool>&,
                      const vector1i&,
                      const vector2d&,
                      const bool&>())
        .def("get_x", &ModelPCAPy::get_x, 
                py::return_value_policy::reference_internal)
        .def("get_fbegin", &ModelPCAPy::get_fbegin, 
                py::return_value_policy::reference_internal)
        .def("get_sbegin", &ModelPCAPy::get_sbegin, 
                py::return_value_policy::reference_internal)
        ;

    py::class_<ModelSinglePy>(m, "PotentialModelSingle")
        .def(py::init<const vector2d&,
                      const vector2d&,
                      const vector1i&,
                      const int&, 
                      const bool&,
                      const vector2d&,
                      const double&,
                      const std::string&,
                      const std::string&,
                      const int&,
                      const int&,
                      const int&,
                      const vector3i&,
                      const vector2i&,
                      const vector2d&,
                      const bool&>())
        .def("get_x", &ModelSinglePy::get_x, 
                py::return_value_policy::reference_internal)
        ;

    py::class_<ModelForChargePy>(m, "PotentialModelCharge")
        .def(py::init<const vector3d&,
                      const vector3d&,
                      const vector2i&,
                      const int&, 
                      const vector2d&,
                      const double&,
                      const std::string&,
                      const std::string&,
                      const int&,
                      const int&,
                      const int&,
                      const vector3i&,
                      const vector2i&,
                      const vector2d&,
                      const vector1i&,
                      const bool&>())
        .def("get_x", &ModelForChargePy::get_x, 
                py::return_value_policy::reference_internal)
        .def("get_cbegin", &ModelForChargePy::get_cbegin, 
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

    py::class_<Ewald>(m, "Ewald")
        .def(py::init<const vector2d&,
                      const vector2d&,
                      const vector1i&,
                      const int&,
                      const double&,
                      const vector2d&,
                      const vector1d&,
                      const double&,
                      const double&,
                      const bool&>())
        .def("get_real_energy", &Ewald::get_real_energy,
            py::return_value_policy::reference_internal)
        .def("get_reciprocal_energy", &Ewald::get_reciprocal_energy,
            py::return_value_policy::reference_internal)
        .def("get_self_energy", &Ewald::get_self_energy,
            py::return_value_policy::reference_internal)
        .def("get_energy", &Ewald::get_energy, 
            py::return_value_policy::reference_internal)
        .def("get_force", &Ewald::get_force_vector1d, 
            py::return_value_policy::reference_internal)
        .def("get_real_force", &Ewald::get_real_force_vector1d, 
            py::return_value_policy::reference_internal)
        .def("get_reciprocal_force", &Ewald::get_reciprocal_force_vector1d, 
            py::return_value_policy::reference_internal)
        .def("get_stress", &Ewald::get_stress_vector1d, 
            py::return_value_policy::reference_internal)
        .def("get_real_stress", &Ewald::get_real_stress_vector1d, 
            py::return_value_policy::reference_internal)
        .def("get_reciprocal_stress", &Ewald::get_reciprocal_stress_vector1d, 
            py::return_value_policy::reference_internal)
        ;

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


