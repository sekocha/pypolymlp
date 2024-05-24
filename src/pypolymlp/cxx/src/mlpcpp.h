/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#ifndef __MLPCPP
#define __MLPCPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <array>
#include <string>
#include <complex>
#include <numeric>
#include <algorithm>

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

#endif
