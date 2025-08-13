/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES_POLYNOMIAL
#define __POLYMLP_FEATURES_POLYNOMIAL

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_model_params.h"
#include "polymlp_products.h"


class FeaturesPoly {

    int n_type, n_variables;
    vector3i prod_features;
    std::vector<MapFromVec> prod_features_map;

    int set_polynomial(const ModelParams& modelp, Maps& maps);
    int set_mappings(Maps& maps);

    public:

    FeaturesPoly();
    FeaturesPoly(const ModelParams& modelp, Maps& maps);
    ~FeaturesPoly();

    void compute_prod_features(const vector1d& features, const int t1, vector1d& vals);

    MapFromVec& get_prod_features_map(const int type1);
    const int get_n_variables() const;

};

#endif
