#include "polymlp_gtinv_data_v2.h"

template<>
const vector2i GtinvDataVer2<2,1>::L_ARRAY_ALL = {
  {1,1}
};

template<>
const vector2d GtinvDataVer2<2,1>::COEFFS_ALL = {
  {
  0.5773502691896258,
  -0.5773502691896258,
  0.5773502691896258
  }
};

template<>
const vector3i GtinvDataVer2<2,1>::M_ARRAY_ALL = {
 {
  {-1,1},
  {0,0},
  {1,-1}
 }
};

template class GtinvDataVer2<2,1>;
