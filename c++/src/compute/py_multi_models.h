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

#ifndef __MULTIPLE_MODELS_PY
#define __MULTIPLE_MODELS_PY

#include <iomanip> 

#include "mlpcpp.h"
#include "compute/neighbor.h"
#include "compute/model.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class MultipleModelsPy {

    Eigen::MatrixXd x_all;
    vector1i xf_begin_dataset, xs_begin_dataset;

    void set_index(const std::vector<int>& n_data_dataset, 
                   const std::vector<bool>& force_dataset,
                   const std::vector<int>& n_atoms_st,
                   std::vector<int>& xf_begin, 
                   std::vector<int>& xs_begin,
                   std::vector<bool>& force, 
                   int& n_row);

    public: 

    MultipleModelsPy(const py::dict& params_dict,
                     const vector3d& axis, 
                     const vector3d& positions_c,
                     const vector2i& types, 
                     const vector1i& n_st_dataset,
                     const std::vector<bool>& force_dataset,
                     const vector1i& n_atoms_all);

    ~MultipleModelsPy();

    Eigen::MatrixXd& get_x();
    const vector1i& get_fbegin() const;
    const vector1i& get_sbegin() const;

};

