/****************************************************************************

        Copyright (C) 2023 Atsuto Seko
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

#ifndef __PYPROPERTIES
#define __PYPROPERTIES

//#include <iomanip> 

#include "mlpcpp.h"
#include "compute/neighbor.h"
#include "compute/model.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyProperties {

    Eigen::VectorXd e_all;
    vector2d f_all;
    Eigen::MatrixXd s_all;

    public: 

    PyProperties(const py::dict& params_dict,
                 const vector1d& coeffs, 
                 const vector3d& axis, 
                 const vector3d& positions_c,
                 const vector2i& types);

    ~PyProperties();

    Eigen::VectorXd& get_e();
    const vector2d& get_f() const;
    Eigen::MatrixXd& get_s();

};

#endif
