/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES
#define __POLYMLP_FEATURES

#include "polymlp_mlpcpp.h"
#include "polymlp_mapping.h"
#include "polymlp_model_params.h"


struct SingleTerm {
    double coeff;
    vector1i nlmtp_keys;
    vector1i type1;
};

typedef std::vector<SingleTerm> SingleFeature;
typedef std::vector<SingleFeature> MultipleFeatures;

typedef std::unordered_map<vector1i,int,HashVI> MapFromVec;

class Features {

    ModelParams modelp;
    Mapping mapping;

//    int n_fn, n_lm, n_lm_all, n_tc, n_nlmtc_all, n_type;

    vector2i type_pairs;

    vector2i feature_combinations;
    MultipleFeatures mfeatures;

    MultipleFeatures set_linear_features_pair();
    MultipleFeatures set_linear_features(const feature_params& fp);

    // not used
    SingleFeature product_features(const SingleFeature& feature1,
                                   const SingleFeature& feature2);

    public:

    Features();
    Features(const feature_params& fp);
    ~Features();

    const MultipleFeatures& get_features() const;

    const vector2i& get_feature_combinations() const;

    const int get_n_features() const;
    const int get_n_feature_combinations() const;
    const int get_n_nlmtc_all() const;
    const int get_n_type() const;
    const Mapping& get_mapping() const;

};

#endif
