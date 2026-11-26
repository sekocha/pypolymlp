/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES_PAIR
#define __POLYMLP_FEATURES_PAIR

//#include <map>

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_mapping.h"
#include "polymlp_model_params.h"


class FeaturesPair {

    int n_type, n_variables;
    vector1i feature_sizes;

    Mapping mapping;
    ModelParams modelp;

    int set_linear_features_pair();
    int set_polynomial();

    public:

    FeaturesPair();
    FeaturesPair(const feature_params& fp);
    ~FeaturesPair();

    void release_memory();
    void compute_features(const vector1d& antp, const int t1, vector1d& vals);

    Maps& get_maps();
    const int get_n_variables() const;

};

#endif
