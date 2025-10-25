/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES_UTILS
#define __POLYMLP_FEATURES_UTILS


#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_model_params.h"


int set_linear_features_pair(Maps& maps);

int set_linear_features_gtinv(
    const feature_params& fp,
    const ModelParams& modelp,
    Maps& maps
);

int get_linear_features_gtinv_with_reps(
    const feature_params& fp,
    const ModelParams& modelp,
    Maps& maps,
    std::vector<MultipleFeatures>& features_for_map
);

int find_local_ids(
    Maps& maps,
    const int type1,
    const int n,
    const vector1i& lm_comb,
    const vector1i& tp_comb,
    vector1i& local_ids
);

#endif
