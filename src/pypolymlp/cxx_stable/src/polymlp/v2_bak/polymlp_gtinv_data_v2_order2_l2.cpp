#include "polymlp_gtinv_data_v2.h"

template<>
const vector2i GtinvDataVer2<2,2>::L_ARRAY_ALL = {
  {2,2}
};

template<>
const vector2d GtinvDataVer2<2,2>::COEFFS_ALL = {
  {
  0.44721359549995804,
  -0.44721359549995787,
  0.4472135954999579,
  -0.4472135954999579,
  0.4472135954999579
  }
};

template<>
const vector3i GtinvDataVer2<2,2>::M_ARRAY_ALL = {
 {
  {-2,2},
  {-1,1},
  {0,0},
  {1,-1},
  {2,-2}
 }
};

template class GtinvDataVer2<2,2>;
