/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/


#include "polymlp_features_polynomial.h"

FeaturesPoly::FeaturesPoly(){}

FeaturesPoly::FeaturesPoly(const ModelParams& modelp, Maps& maps){

    n_type = maps.get_n_type();
    set_polynomial(modelp, maps);
    set_mappings(maps);
}

FeaturesPoly::~FeaturesPoly(){}

int FeaturesPoly::set_polynomial(const ModelParams& modelp, Maps& maps){

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

int FeaturesPoly::set_mappings(Maps& maps){

    prod_features.resize(n_type);
    prod_features_map.resize(n_type);
    for (int t1 = 0; t1 < n_type; ++t1){
        std::set<vector1i> nonequiv;
        auto& maps_type = maps.maps_type[t1];
        for (const auto& term: maps_type.polynomial){
            for (size_t i = 0; i < term.local_ids.size(); ++i){
                vector1i keys = erase_a_key(term.local_ids, i);
                nonequiv.insert(keys);
            }
        }
        convert_set_to_mappings(nonequiv, prod_features_map[t1], prod_features[t1]);
    }
    return 0;
}


void FeaturesPoly::compute_prod_features(
    const vector1d& features,
    const int type1,
    vector1d& vals
){
    compute_products<double>(prod_features[type1], features, vals);
}

MapFromVec& FeaturesPoly::get_prod_features_map(const int type1){
    return prod_features_map[type1];
}

const int FeaturesPoly::get_n_variables() const {
    return n_variables;
}
