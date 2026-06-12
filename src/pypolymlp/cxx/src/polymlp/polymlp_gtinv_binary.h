/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_GTINV_DATA_BINARY
#define __POLYMLP_GTINV_DATA_BINARY

#include <cstdint>
#include <cstring>
#include "polymlp_mlpcpp.h"

typedef std::vector<std::vector<int32_t> > vector2i32;
typedef std::vector<std::vector<std::vector<int32_t> > > vector3i32;

int32_t read_int32(std::ifstream &ifs);
double read_double(std::ifstream &ifs);

vector2i32 read_2d_int(std::ifstream &ifs);
vector3i32 read_3d_int(std::ifstream &ifs);
vector2d read_2d_double(std::ifstream &ifs);

#endif
