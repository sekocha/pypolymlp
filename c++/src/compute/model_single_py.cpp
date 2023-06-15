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

	    Main program for calculating descriptors using Python

*****************************************************************************/

#include "model_single_py.h"

ModelSinglePy::ModelSinglePy(const vector2d& axis,
                             const vector2d& positions_c,
                             const vector1i& types,
                             const int& n_type,
                             const bool& force,
                             const vector2d& params,
                             const double& cutoff,
                             const std::string& pair_type,
                             const std::string& des_type,
                             const int& model_type,
                             const int& maxp,
                             const int& maxl,
                             const vector3i& lm_array,
                             const vector2i& l_comb,
                             const vector2d& lm_coeffs,
                             const bool& element_swap){

    fp = {n_type,
          force,
          params,
          cutoff,
          pair_type,
          des_type,
          model_type,
          maxp,
          maxl,
          lm_array,
          l_comb,
          lm_coeffs};

    Neighbor neigh(axis, positions_c, types, fp.n_type, fp.cutoff);
    Model mod(neigh.get_dis_array(), 
              neigh.get_diff_array(),
              neigh.get_atom2_array(), 
              types, 
              fp, 
              element_swap);
    xe = mod.get_xe_sum();
}

ModelSinglePy::~ModelSinglePy(){}

const vector1d& ModelSinglePy::get_x() const{ return xe; }

//vector1d ModelSinglePy::compute
//
//    Neighbor neigh(axis, positions_c, types, fp.n_type, fp.cutoff);
//    Model mod(neigh.get_dis_array(), neigh.get_diff_array(),
//            neigh.get_atom2_array(), types, fp);
//    return mod.get_xe_sum();
//}


