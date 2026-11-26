/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        SingleTerm: [coeff,[(n1,l1,m1,tc1), (n2,l2,m2,tc2), ...]]
            (n1,l1,m1,tc1) is represented with nlmtc_key.

        SingleFeature: [SingleTerms1, SingleTerms2, ...]

        MultipleFeaturesPair: [SingleFeature1, SingleFeature2, ...]

*****************************************************************************/

#include "polymlp_features_pair.h"

const int threshold_prod = 1000;


FeaturesPair::FeaturesPair(){}

FeaturesPair::FeaturesPair(const feature_params& fp){

    n_type = fp.n_type;
    mapping = Mapping(fp);
    auto& maps = mapping.get_maps();
    modelp = ModelParams(fp, maps);

    set_linear_features_pair();
    set_polynomial();
}

FeaturesPair::~FeaturesPair(){}

int FeaturesPair::set_linear_features_pair(){

    auto& maps = mapping.get_maps();
    for (const auto& ntp: maps.ntp_attrs){
        const auto& types = maps.tp_to_types[ntp.tp];
        const int type1 = types[0];
        const int type2 = types[1];

        vector1i key = {ntp.global_id, type1};
        const int local_id = maps.ntp_global_to_iloc[key];
        maps.maps_type[type1].features_pair.emplace_back(local_id);
        if (type1 != type2){
            key = {ntp.global_id, type2};
            local_id = maps.ntp_global_to_iloc[key];
            maps.maps_type[type2].features_pair.emplace_back(local_id);
        }
    }

    //for (int type1 = 0; type1 < n_type; ++type1){
    //    const auto& features1 = maps.maps_type[type1].features_pair;
    //    feature_sizes.emplace_back(features1.size());
    //}

    return 0;
}

int FeaturesPair::set_polynomial(){

    auto& maps = mapping.get_maps();
    const int n_linear_features = modelp.get_n_linear_features();
    const auto& comb2 = modelp.get_comb2();
    const auto& comb3 = modelp.get_comb3();
    n_variables = n_linear_features + comb2.size() + comb3.size();

    int c1, c2, c3, begin;
    for (int type1 = 0; type1 < n_type; ++type1){
        auto& polynomial = maps.maps_type[type1].polynomial;
        int local_id(0);
        std::unordered_map<int,int> map1;
        for (const auto& i: modelp.get_comb1_indices(type1)){
            PolynomialTerm pterm = {i, vector1i{local_id}};
            polynomial.emplace_back(pterm);
            map1[i] = local_id;
            ++local_id;
        }
        begin = n_linear_features;
        for (const auto& i: modelp.get_comb2_indices(type1)){
            c1 = comb2[i][0], c2 = comb2[i][1];
            PolynomialTerm pterm = {begin + i, vector1i{map1[c1], map1[c2]}};
            polynomial.emplace_back(pterm);
        }
        begin = n_linear_features + comb2.size();
        for (const auto& i: modelp.get_comb3_indices(type1)){
            c1 = comb3[i][0], c2 = comb3[i][1], c3 = comb3[i][2];
            PolynomialTerm pterm = {
                begin + i, vector1i{map1[c1], map1[c2], map1[c3]}
            };
            polynomial.emplace_back(pterm);
        }
    }
    return 0;
}


// void FeaturesPair::release_memory(){
//     auto& maps = mapping.get_maps();
//     maps.clear();
//     prod_map_deriv_from_keys = {};
// }


void FeaturesPair::compute_features(
    const vector1d& antp,
    const int type1,
    vector1d& feature_values
){
    auto& maps = mapping.get_maps();
    const auto& features1 = maps.maps_type[type1].features_pair;

    feature_values = vector1d({});
    for (const auto& local_id; features1){
        feature_values.emplace_back(antp[local_id]);
    }
}

void FeaturesPair::compute_prod_antp_deriv(
    const vector1d& antp,
    const int type1,
    vector1d& prod_antp_deriv
){
//    compute_products<double>(prod_deriv[type1], antp, prod_antp_deriv);
}

//
//void FeaturesPair::compute_prod_features(
//    const vector1d& features,
//    const int type1,
//    vector1d& values
//){
//    poly.compute_prod_features(features, type1, values);
//}


Maps& FeaturesPair::get_maps() { return mapping.get_maps(); }


//MapFromVec& FeaturesPair::get_prod_map_deriv(const int type1){
//    return prod_map_deriv_from_keys[type1];
//}
//
//
//MapFromVec& FeaturesPair::get_prod_features_map(const int type1){
//    return poly.get_prod_features_map(type1);
//}
//
const int FeaturesPair::get_n_variables() const {
    return n_variables;
}
