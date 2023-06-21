/****************************************************************************

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
