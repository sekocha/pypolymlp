/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/


#include "features.h"

FunctionFeatures::FunctionFeatures(){}

FunctionFeatures::FunctionFeatures(
    const feature_params& fp,
    const ModelParams& modelp,
    const Features& f_obj
){

    n_type = f_obj.get_n_type();

    if (fp.des_type == "gtinv"){
        lm_map = f_obj.get_lm_map();
        nlmtc_map = f_obj.get_nlmtc_map();
        nlmtc_map_no_conjugate = f_obj.get_nlmtc_map_no_conjugate();

        n_nlmtc_all = f_obj.get_n_nlmtc_all();

        prod_map.resize(n_type);
        prod_map_from_keys.resize(n_type);

        prod_map_deriv.resize(n_type);
        prod_map_deriv_from_keys.resize(n_type);

        linear_features.resize(n_type);
        linear_features_deriv.resize(n_type);

        set_mapping_prod(f_obj);
        set_features_using_mappings(f_obj);
    }

    set_polynomials(modelp);

}

FunctionFeatures::~FunctionFeatures(){}

void FunctionFeatures::set_mapping_prod(const Features& f_obj){

    std::vector<std::set<vector1i> >
        nonequiv_keys(n_type), nonequiv_deriv_keys(n_type);

    const auto& mfeatures = f_obj.get_features();
    for (const auto& sfeature: mfeatures){
        const auto type1 = sfeature[0].type1;
        for (const auto& sterm: sfeature){
            for (const auto& t1: type1){
                nonequiv_keys[t1].insert(sterm.nlmtc_keys);
            }
            for (size_t i = 0; i < sterm.nlmtc_keys.size(); ++i){
                const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                for (const auto& t1: type1){
                    nonequiv_deriv_keys[t1].insert(keys);
                }
            }
        }
    }

    for (int t1 = 0; t1 < n_type; ++t1){
        nonequiv_set_to_mappings(nonequiv_keys[t1],
                                 prod_map_from_keys[t1],
                                 prod_map[t1]);
        nonequiv_set_to_mappings(nonequiv_deriv_keys[t1],
                                 prod_map_deriv_from_keys[t1],
                                 prod_map_deriv[t1]);
    }

}

void FunctionFeatures::set_features_using_mappings(const Features& f_obj){

    const auto& mfeatures = f_obj.get_features();
    for (int t1 = 0; t1 < n_type; ++t1){
        linear_features[t1].resize(mfeatures.size());
        linear_features_deriv[t1].resize(mfeatures.size());
    }

    int idx(0);
    for (const auto& sfeature: mfeatures){
        const auto type1 = sfeature[0].type1;
        for (const auto& t1: type1){
            std::unordered_map<int, double> sfeature_map;
            for (const auto& sterm: sfeature){
                const int prod_key = prod_map_from_keys[t1][sterm.nlmtc_keys];
                if (sfeature_map.count(prod_key) == 0){
                    sfeature_map[prod_key] = sterm.coeff;
                }
                else {
                    sfeature_map[prod_key] += sterm.coeff;
                }
            }
            for (const auto& sterm: sfeature_map){
                FeatureSingleTerm fsterm = {sterm.second, sterm.first};
                linear_features[t1][idx].emplace_back(fsterm);
            }
        }

        for (const auto& t1: type1){
            std::unordered_map<vector1i, double, HashVI> sfeature_map;
            for (const auto& sterm: sfeature){
                for (size_t i = 0; i < sterm.nlmtc_keys.size(); ++i){
                    const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                    const int prod_key = prod_map_deriv_from_keys[t1][keys];
                    const int nlmtc_key = sterm.nlmtc_keys[i];
                    vector1i map_key = {prod_key, nlmtc_key};
                    if (sfeature_map.count(map_key) == 0){
                        sfeature_map[map_key] = sterm.coeff;
                    }
                    else {
                        sfeature_map[map_key] += sterm.coeff;
                    }
                }
            }
            for (const auto& sterm: sfeature_map){
                FeatureSingleTermDeriv fsterm
                    = {sterm.second, sterm.first[0], sterm.first[1]};
                linear_features_deriv[t1][idx].emplace_back(fsterm);
            }
        }
        ++idx;
    }
    //sort_linear_features_deriv();
}

void FunctionFeatures::sort_linear_features_deriv(){

    // sorted by prod_key and then by prod_features_key
    for (int t1 = 0; t1 < n_type; ++t1){
        for (auto& sfeature: linear_features_deriv[t1]){
            std::sort(sfeature.begin(), sfeature.end(),
                    [](const FeatureSingleTermDeriv& lhs,
                        const FeatureSingleTermDeriv& rhs){
                    if (lhs.nlmtc_key != rhs.nlmtc_key){
                        return lhs.nlmtc_key < rhs.nlmtc_key;
                    }
                    else {
                        return lhs.prod_key < rhs.prod_key;
                    }
                    });
        }
    }
}


void FunctionFeatures::set_features_using_mappings_simple(const Features& f_obj){

    const auto& mfeatures = f_obj.get_features();
    for (int t1 = 0; t1 < n_type; ++t1){
        linear_features[t1].resize(mfeatures.size());
        linear_features_deriv[t1].resize(mfeatures.size());
    }

    int idx(0);
    for (const auto& sfeature: mfeatures){
        const auto type1 = sfeature[0].type1;
        for (const auto& sterm: sfeature){
            for (const auto& t1: type1){
                const int prod_key = prod_map_from_keys[t1][sterm.nlmtc_keys];
                FeatureSingleTerm fsterm = {sterm.coeff, prod_key};
                linear_features[t1][idx].emplace_back(fsterm);
            }
        }

        for (const auto& sterm: sfeature){
            for (size_t i = 0; i < sterm.nlmtc_keys.size(); ++i){
                const int nlmtc_key = sterm.nlmtc_keys[i];
                const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                for (const auto& t1: type1){
                    const int prod_key = prod_map_deriv_from_keys[t1][keys];
                    FeatureSingleTermDeriv fsterm
                            = {sterm.coeff, prod_key, nlmtc_key};
                    linear_features_deriv[t1][idx].emplace_back(fsterm);
                }
            }
        }
        ++idx;
    }
}

void FunctionFeatures::nonequiv_set_to_mappings(
    const std::set<vector1i>& nonequiv_keys,
    ProdMapFromKeys& map_from_keys,
    vector2i& map
){

    map = vector2i(nonequiv_keys.begin(), nonequiv_keys.end());
    std::sort(map.begin(), map.end());

    int i(0);
    for (const auto& keys: map){
        map_from_keys[keys] = i;
        ++i;
    }
}


vector1i FunctionFeatures::erase_a_key(const vector1i& original, const int idx){
    vector1i keys = original;
    keys.erase(keys.begin() + idx);
    std::sort(keys.begin(), keys.end());
    return keys;
}

/*
void FunctionFeatures::print_keys(const vector1i& keys){
    for (const auto& k: keys)
        std::cout << k << " ";
    std::cout << std::endl;
}
*/

void FunctionFeatures::set_polynomials(const ModelParams& modelp){

    polynomials1.resize(n_type);
    polynomials2.resize(n_type);
    polynomials3.resize(n_type);

    const int n_linear_features = modelp.get_n_des();
    const auto& comb2 = modelp.get_comb2();
    const auto& comb3 = modelp.get_comb3();

    int c1, c2, c3, begin;
    for (int type1 = 0; type1 < n_type; ++type1){
        int tlocal_id(0);
        std::unordered_map<int,int> map1;
        for (const auto& i: modelp.get_comb1_indices(type1)){
            PolynomialTerm pterm = {i, vector1i{}};
            polynomials1[type1].emplace_back(pterm);
            map1[i] = tlocal_id;
            ++tlocal_id;
        }
        begin = n_linear_features;
        for (const auto& i: modelp.get_comb2_indices(type1)){
            c1 = comb2[i][0], c2 = comb2[i][1];
            PolynomialTerm pterm = {begin + i, vector1i{map1[c1], map1[c2]}};
            polynomials2[type1].emplace_back(pterm);
        }
        begin = n_linear_features + comb2.size();
        for (const auto& i: modelp.get_comb3_indices(type1)){
            c1 = comb3[i][0], c2 = comb3[i][1], c3 = comb3[i][2];
            PolynomialTerm pterm = {
                begin + i, vector1i{map1[c1], map1[c2], map1[c3]}
            };
            polynomials3[type1].emplace_back(pterm);
        }
    }
}

const std::vector<lmAttribute>& FunctionFeatures::get_lm_map() const {
    return lm_map;
}
const std::vector<nlmtcAttribute>&
FunctionFeatures::get_nlmtc_map_no_conjugate() const{
    return nlmtc_map_no_conjugate;
}
const std::vector<nlmtcAttribute>& FunctionFeatures::get_nlmtc_map() const {
    return nlmtc_map;
}
const int FunctionFeatures::get_n_nlmtc_all() const {
    return n_nlmtc_all;
}


const vector2i& FunctionFeatures::get_prod_map(const int t) const {
    return prod_map[t];
}
const vector2i& FunctionFeatures::get_prod_map_deriv(const int t) const {
    return prod_map_deriv[t];
}

const FeatureVector&
FunctionFeatures::get_linear_features(const int t) const {
    return linear_features[t];
}
const FeatureVectorDeriv&
FunctionFeatures::get_linear_features_deriv(const int t) const {
    return linear_features_deriv[t];
}

const Polynomial& FunctionFeatures::get_polynomial1(const int t) const {
    return polynomials1[t];
}
const Polynomial& FunctionFeatures::get_polynomial2(const int t) const {
    return polynomials2[t];
}
const Polynomial& FunctionFeatures::get_polynomial3(const int t) const {
    return polynomials3[t];
}
