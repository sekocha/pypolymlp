#include "polymlp_gtinv_data_v2.h"

template<>
const vector2i GtinvDataVer2<2,4>::L_ARRAY_ALL = {
  {4,4}
};

template<>
const vector2d GtinvDataVer2<2,4>::COEFFS_ALL = {
  {
  0.33333333333333354,
  -0.333333333333333,
  0.3333333333333333,
  -0.3333333333333333,
  0.3333333333333333,
  -0.33333333333333326,
  0.33333333333333326,
  -0.33333333333333326,
  0.33333333333333326
  }
};

template<>
const vector3i GtinvDataVer2<2,4>::M_ARRAY_ALL = {
 {
  {-4,4},
  {-3,3},
  {-2,2},
  {-1,1},
  {0,0},
  {1,-1},
  {2,-2},
  {3,-3},
  {4,-4}
 }
};

template class GtinvDataVer2<2,4>;
