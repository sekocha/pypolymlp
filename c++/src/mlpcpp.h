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

*****************************************************************************/

#ifndef __MLPCPP
#define __MLPCPP

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <complex>
#include <numeric>
#include <algorithm>

#include <armadillo>

#include "polymlp/polymlp_mlpcpp.h"

using vector1i = std::vector<int>;
using vector2i = std::vector<vector1i>;
using vector3i = std::vector<vector2i>;
using vector4i = std::vector<vector3i>;
using vector1d = std::vector<double>;
using vector2d = std::vector<vector1d>;
using vector3d = std::vector<vector2d>;
using vector4d = std::vector<vector3d>;
using vector5d = std::vector<vector4d>;

using dc = std::complex<double>;
using vector1dc = std::vector<dc>;
using vector2dc = std::vector<vector1dc>;
using vector3dc = std::vector<vector2dc>;
using vector4dc = std::vector<vector3dc>;

using vector1mat = std::vector<arma::mat>;
using vector2mat = std::vector<vector1mat>;
using vector3mat = std::vector<vector2mat>;
using vector1vec = std::vector<arma::vec>;
using vector2vec = std::vector<vector1vec>;
using vector3vec = std::vector<vector2vec>;

using vector1cx_vec = std::vector<arma::cx_vec>;
using vector2cx_vec = std::vector<vector1cx_vec>;
using vector3cx_vec = std::vector<vector2cx_vec>;
using vector1cx_mat = std::vector<arma::cx_mat>;
using vector2cx_mat = std::vector<vector1cx_mat>;
using vector3cx_mat = std::vector<vector2cx_mat>;

#endif
