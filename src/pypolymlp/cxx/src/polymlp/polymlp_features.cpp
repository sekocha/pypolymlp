/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        SingleTerm: [coeff,[(n1,l1,m1,tc1), (n2,l2,m2,tc2), ...]]
            (n1,l1,m1,tc1) is represented with nlmtc_key.

        SingleFeature: [SingleTerms1, SingleTerms2, ...]

        MultipleFeatures: [SingleFeature1, SingleFeature2, ...]

*****************************************************************************/

#include "polymlp_features.h"

Features::Features(){}

Features::Features(const feature_params& fp){

    std::cout << "Mapping" << std::endl;
    mapping = Mapping(fp);
    std::cout << "Mapping: finished" << std::endl;
    modelp = ModelParams(fp, mapping);
    std::cout << "ModelParams: finished" << std::endl;
    n_type = fp.n_type;
//    type_pairs = mapping.get_type_pairs();

    MultipleFeatures mfeatures1;
    if (fp.feature_type == "pair") mfeatures = set_linear_features_pair();
    else if (fp.feature_type == "gtinv") mfeatures = set_linear_features(fp);

    // this part can be revised in a recursive form
    for (size_t i = 0; i < mfeatures.size(); ++i){
        const vector1i c = {int(i)};
        feature_combinations.emplace_back(c);
    }
    const auto& comb2 = modelp.get_comb2();
    const auto& comb3 = modelp.get_comb3();
    for (const auto& c: comb2) feature_combinations.emplace_back(c);
    for (const auto& c: comb3) feature_combinations.emplace_back(c);

}

Features::~Features(){}

MultipleFeatures Features::set_linear_features_pair(){
/*
    // TODO: MUST BE REVISED.
    // The order of tc and n must be fixed.
    std::vector<SingleFeature> feature_array;
    for (int tc = 0; tc < n_tc; ++tc){
        vector1i type1;
        for (int t1 = 0; t1 < n_type; ++t1){
            const auto& tp_t1 = type_pairs[t1];
            if (std::find(tp_t1.begin(), tp_t1.end(), tc) != tp_t1.end()){
                type1.emplace_back(t1);
            }
        }
        for (int n = 0; n < n_fn; ++n){
            SingleFeature feature;
            SingleTerm single;
            single.coeff = 1.0;
            single.type1 = type1;
            const int key = mapping_ntc_to_key(tc, n);
            single.nlmtc_keys.emplace_back(key);
            feature.emplace_back(single);
            feature_array.emplace_back(feature);
        }
    }
    return feature_array;
    */
    std::vector<SingleFeature> feature_array;
    return feature_array;
}

MultipleFeatures Features::set_linear_features(const feature_params& fp){

    const vector3i& lm_array = fp.lm_array;
    const vector2d& lm_coeffs = fp.lm_coeffs;
    const auto& tp_combs = modelp.get_tp_combs();
    MapFromVec map_nlmtp_to_key = mapping.get_nlmtp_to_key();

    std::vector<SingleFeature> feature_array;
    for (const auto& linear: modelp.get_linear_terms()){
        const int n = linear.n;
        const int n_id = linear.n_id;
        const int lm_comb_id = linear.lm_comb_id;
        const int tp_comb_id = linear.tp_comb_id;
        const int order = linear.order;
        const auto& tp_comb = tp_combs[order][tp_comb_id];
        const auto& type1 = linear.type1;

        const auto& lm_list = lm_array[lm_comb_id];
        const auto& coeff_list = lm_coeffs[lm_comb_id];

        SingleFeature feature;
        for (size_t i = 0; i < lm_list.size(); ++i){
            SingleTerm single;
            single.coeff = coeff_list[i];
            single.type1 = type1;
            for (size_t j = 0; j < lm_list[i].size(); ++j){
                const auto lm = lm_list[i][j];
                const auto tp = tp_comb[j];
                int key = map_nlmtp_to_key[vector1i({n, lm, tp})];
                single.nlmtp_keys.emplace_back(key);
            }
            std::sort(single.nlmtp_keys.begin(), single.nlmtp_keys.end());
            feature.emplace_back(single);
        }
        feature_array.emplace_back(feature);
    }
    return feature_array;
}

const int Features::get_n_type() const { return n_type; }
const int Features::get_n_features() const { return mfeatures.size(); }
const int Features::get_n_feature_combinations() const {
    return feature_combinations.size();
}
const MultipleFeatures& Features::get_features() const {
    return mfeatures;
}
const vector2i& Features::get_feature_combinations() const {
    return feature_combinations;
}
const Mapping& Features::get_mapping() const {
    return mapping;
}
const ModelParams& Features::get_model_params() const {
    return modelp;
}
