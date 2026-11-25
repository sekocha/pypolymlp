#include "polymlp_gtinv_data_v2.h"

template<>
const vector2i GtinvDataVer2<3,0>::L_ARRAY_ALL = {
  {0,0,0}
};

template<>
const vector2d GtinvDataVer2<3,0>::COEFFS_ALL = {
  {
  1.0
  }
};

template<>
const vector3i GtinvDataVer2<3,0>::M_ARRAY_ALL = {
 {
  {0,0,0}
 }
};

template class GtinvDataVer2<3,0>;
