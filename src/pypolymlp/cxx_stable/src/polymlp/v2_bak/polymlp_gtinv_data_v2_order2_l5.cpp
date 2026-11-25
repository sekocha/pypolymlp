#include "polymlp_gtinv_data_v2.h"

template<>
const vector2i GtinvDataVer2<2,5>::L_ARRAY_ALL = {
  {5,5}
};

template<>
const vector2d GtinvDataVer2<2,5>::COEFFS_ALL = {
  {
  0.3015113445777637,
  -0.3015113445777635,
  0.30151134457776363,
  -0.30151134457776363,
  0.30151134457776363,
  -0.3015113445777636,
  0.3015113445777636,
  -0.3015113445777636,
  0.3015113445777636,
  -0.30151134457776363,
  0.30151134457776363
  }
};

template<>
const vector3i GtinvDataVer2<2,5>::M_ARRAY_ALL = {
 {
  {-5,5},
  {-4,4},
  {-3,3},
  {-2,2},
  {-1,1},
  {0,0},
  {1,-1},
  {2,-2},
  {3,-3},
  {4,-4},
  {5,-5}
 }
};

template class GtinvDataVer2<2,5>;
