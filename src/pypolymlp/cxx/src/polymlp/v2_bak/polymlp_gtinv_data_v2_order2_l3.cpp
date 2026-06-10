#include "polymlp_gtinv_data_v2.h"

template<>
const vector2i GtinvDataVer2<2,3>::L_ARRAY_ALL = {
  {3,3}
};

template<>
const vector2d GtinvDataVer2<2,3>::COEFFS_ALL = {
  {
  0.3779644730092275,
  -0.37796447300922675,
  0.3779644730092271,
  -0.3779644730092271,
  0.3779644730092271,
  -0.37796447300922714,
  0.37796447300922714
  }
};

template<>
const vector3i GtinvDataVer2<2,3>::M_ARRAY_ALL = {
 {
  {-3,3},
  {-2,2},
  {-1,1},
  {0,0},
  {1,-1},
  {2,-2},
  {3,-3}
 }
};

template class GtinvDataVer2<2,3>;
