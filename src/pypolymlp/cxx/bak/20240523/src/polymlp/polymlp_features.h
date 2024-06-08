/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES
#define __POLYMLP_FEATURES

#include "polymlp_mlpcpp.h"
#include "polymlp_model_params.h"

struct SingleTerm {
    double coeff;
    vector1i nlmtc_keys;
    vector1i type1;
};

struct lmAttribute {
    int l;
    int m;
    int ylmkey;
    bool conj;
    double cc_coeff;
    double sign_j;
};

struct nlmtcAttribute {
    int n;
    lmAttribute lm;
    int tc;
    int nlmtc_key;
    int conj_key;
    int nlmtc_noconj_key;
};

struct ntcAttribute {
    int n;
    int tc;
    int ntc_key;
};

typedef std::vector<SingleTerm> SingleFeature;
typedef std::vector<SingleFeature> MultipleFeatures;

class Features {

    int n_fn, n_lm, n_lm_all, n_tc, n_nlmtc_all, n_type;
    std::vector<lmAttribute> lm_map;
    std::vector<nlmtcAttribute> nlmtc_map_no_conjugate, nlmtc_map;
    std::vector<ntcAttribute> ntc_map;
    vector3i type_comb_pair;

    vector2i feature_combinations;
    MultipleFeatures mfeatures;

    MultipleFeatures set_linear_features_pair();
    MultipleFeatures set_linear_features(const feature_params& fp,
                                         const ModelParams& modelp);

    // for des_type == pair
    int mapping_ntc_to_key(const int tc, const int n);
    void set_mapping_ntc();

    // for des_type == gtinv
    int mapping_nlmtc_to_key(const int tc, const int lm, const int n);
    void set_mapping_lm(const int maxl);
    void set_mapping_nlmtc();

    // not used
    SingleFeature product_features(const SingleFeature& feature1,
                                   const SingleFeature& feature2);

    public:

    Features();
    Features(const feature_params& fp, const ModelParams& modelp);
    ~Features();

    const MultipleFeatures& get_features() const;

    // for des_type == pair
    const std::vector<ntcAttribute>& get_ntc_map() const;

    // for des_type == gtinv
    const std::vector<lmAttribute>& get_lm_map() const;
    const std::vector<nlmtcAttribute>& get_nlmtc_map_no_conjugate() const;
    const std::vector<nlmtcAttribute>& get_nlmtc_map() const;

    const vector2i& get_feature_combinations() const;

    const int get_n_features() const;
    const int get_n_feature_combinations() const;
    const int get_n_nlmtc_all() const;
    const int get_n_type() const;

};

#endif
