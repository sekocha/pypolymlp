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

    n_type = fp.n_type;
    mapping = Mapping(fp);
    modelp = ModelParams(fp, mapping);

    MultipleFeatures mfeatures1;
    if (fp.feature_type == "pair") mfeatures = set_linear_features_pair();
    else if (fp.feature_type == "gtinv") mfeatures = set_linear_features(fp);

    for (size_t i = 0; i < mfeatures.size(); ++i)
        feature_combinations.emplace_back(vector1i({int(i)}));
    for (const auto& c: modelp.get_comb2()) feature_combinations.emplace_back(c);
    for (const auto& c: modelp.get_comb3()) feature_combinations.emplace_back(c);
}

Features::~Features(){}

MultipleFeatures Features::set_linear_features_pair(){

    const auto& n_type_pairs = mapping.get_n_type_pairs();
    const auto& type_pairs = mapping.get_type_pairs();
    const auto& ntp_attrs = mapping.get_ntp_attrs();

    vector2i type1_map;
    for (int tp = 0; tp < n_type_pairs; ++tp){
        vector1i type1;
        for (int t1 = 0; t1 < n_type; ++t1){
            const auto& tp_t1 = type_pairs[t1];
            if (std::find(tp_t1.begin(), tp_t1.end(), tp) != tp_t1.end()){
                type1.emplace_back(t1);
            }
        }
        type1_map.emplace_back(type1);
    }

    std::vector<SingleFeature> feature_array;
    for (const auto& ntp: ntp_attrs){
        SingleFeature feature;
        SingleTerm single;
        single.coeff = 1.0;
        single.type1 = type1_map[ntp.tp];
        single.nlmtp_keys.emplace_back(ntp.ntp_key);
        feature.emplace_back(single);
        feature_array.emplace_back(feature);
    }
    return feature_array;
}

MultipleFeatures Features::set_linear_features(const feature_params& fp){

    const vector3i& lm_array = fp.lm_array;
    const vector2d& lm_coeffs = fp.lm_coeffs;
    const auto& tp_combs = modelp.get_tp_combs();
    MapFromVec map_nlmtp_to_key = mapping.get_nlmtp_to_key();

    std::vector<SingleFeature> feature_array;
    for (const auto& linear: modelp.get_linear_terms()){
        const auto& tp_comb = tp_combs[linear.order][linear.tp_comb_id];
        const auto& lm_list = lm_array[linear.lm_comb_id];
        const auto& coeff_list = lm_coeffs[linear.lm_comb_id];

        SingleFeature feature;
        for (size_t i = 0; i < lm_list.size(); ++i){
            SingleTerm single;
            single.coeff = coeff_list[i];
            single.type1 = linear.type1;
            for (size_t j = 0; j < lm_list[i].size(); ++j){
                const auto lm = lm_list[i][j];
                const auto tp = tp_comb[j];
                int key = map_nlmtp_to_key[vector1i({linear.n, lm, tp})];
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
const Mapping& Features::get_mapping() const {
    return mapping;
}
const ModelParams& Features::get_model_params() const {
    return modelp;
}
const int Features::get_n_feature_combinations() const {
    return feature_combinations.size();
}
const MultipleFeatures& Features::get_features() const {
    return mfeatures;
}
const vector2i& Features::get_feature_combinations() const {
    return feature_combinations;
}
