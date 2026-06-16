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

    py::class_<PyHybridModel>(m, "PotentialHybridModel")
        .def(py::init<const std::vector<py::dict>&,
                      const vector3d&,
                      const vector3d&,
                      const vector2i&,
                      const vector1i&,
                      const std::vector<bool>&,
                      const vector1i&>())
        .def("get_x", &PyHybridModel::get_x,
                py::return_value_policy::reference_internal)
        .def("get_fbegin", &PyHybridModel::get_fbegin,
                py::return_value_policy::reference_internal)
        .def("get_sbegin", &PyHybridModel::get_sbegin,
                py::return_value_policy::reference_internal)
        .def("get_cumulative_n_features",
                &PyHybridModel::get_cumulative_n_features,
                py::return_value_policy::reference_internal)
        .def("get_n_data", &PyHybridModel::get_n_data,
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
        .def(py::init<const int,
                      const vector1i&,
                      const int>())
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

    py::class_<NeighborHalf>(m, "NeighborHalf")
        .def(py::init<const vector2d&,
                      const vector2d&,
                      const vector1i&,
                      const double&,
                      const bool>())
        .def("get_differences", &NeighborHalf::get_diff_list,
                py::return_value_policy::reference_internal)
        .def("get_neighbor_indices", &NeighborHalf::get_half_list,
                py::return_value_policy::reference_internal)
        ;

    py::class_<feature_params>(m, "FeatureParams")
        .def(py::init<>())
        .def_readwrite("n_type", &feature_params::n_type)
        .def_readwrite("force", &feature_params::force)
        .def_readwrite("params", &feature_params::params)
        .def_readwrite("params_conditional", &feature_params::params_conditional)
        .def_readwrite("cutoff", &feature_params::cutoff)
        .def_readwrite("pair_type", &feature_params::pair_type)
        .def_readwrite("feature_type", &feature_params::feature_type)
        .def_readwrite("model_type", &feature_params::model_type)
        .def_readwrite("maxp", &feature_params::maxp)
        .def_readwrite("maxl", &feature_params::maxl)
        .def_readwrite("lm_array", &feature_params::lm_array)
        .def_readwrite("l_comb", &feature_params::l_comb)
        .def_readwrite("lm_coeffs", &feature_params::lm_coeffs)
        ;

    m.def("get_fn",
        [](const double dis,
           const feature_params& fp,
           const vector2d& params){
            vector1d fn;
            vector1d fn_d;
            get_fn_(dis, fp, params, fn, fn_d);
            return py::make_tuple(fn, fn_d);
        },
        py::arg("dis"),
        py::arg("fp"),
        py::arg("params")
    );

    m.def("get_ylm",
        [](double r,
           double polar,
           double azimuthal,
           int lmax){
            vector1dc ylm;
            vector1dc ylm_dx;
            vector1dc ylm_dy;
            vector1dc ylm_dz;

            get_ylm_(
                r,
                polar,
                azimuthal,
                lmax,
                ylm,
                ylm_dx,
                ylm_dy,
                ylm_dz);

            return py::make_tuple(
                ylm,
                ylm_dx,
                ylm_dy,
                ylm_dz);
        },
        py::arg("r"),
        py::arg("polar"),
        py::arg("azimuthal"),
        py::arg("lmax")
    );
}
