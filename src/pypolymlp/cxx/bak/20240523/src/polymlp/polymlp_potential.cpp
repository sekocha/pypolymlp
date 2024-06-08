/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        SingleTerm: [coeff,[(n1,l1,m1,tc1), (n2,l2,m2,tc2), ...]]
            (n1,l1,m1,tc1) is represented with nlmtc_key.

        SingleFeature: [SingleTerm1, SingleTerm2, ...]

        MultipleFeatures: [SingleFeature1, SingleFeature2, ...]

        PotentialModel: [PotentialTerm1, PotentialTerm2, ...]

        PotentialTerm: anlmtc[head_key] * prod_anlmtc[prod_key]
                                        * feature[feature_key1]
                                        * feature[feature_key2]
                                        * ...

*****************************************************************************/

#include "polymlp_potential.h"

Potential::Potential(){}

Potential::Potential(const Features& f_obj, const vector1d& pot){

    lm_map = f_obj.get_lm_map();
    nlmtc_map = f_obj.get_nlmtc_map();
    nlmtc_map_no_conjugate = f_obj.get_nlmtc_map_no_conjugate();
    ntc_map = f_obj.get_ntc_map();
    n_nlmtc_all = f_obj.get_n_nlmtc_all();
    n_type = f_obj.get_n_type();

    int size;
    if (ntc_map.size() > 0){
        eliminate_conj = false;
        separate_erased = false;
        size = ntc_map.size();
    }
    else {
        eliminate_conj = true;
        separate_erased = true;
        if (eliminate_conj == true) size = nlmtc_map_no_conjugate.size();
        else size = n_nlmtc_all;
    }
    potential_model_each_key = PotentialModelEachKey(n_type);
    for (int t1 = 0; t1 < n_type; ++t1)
        potential_model_each_key[t1].resize(size);

    prod_map.resize(n_type);
    prod_map_from_keys.resize(n_type);
    prod_map_erased.resize(n_type);
    prod_map_erased_from_keys.resize(n_type);
    prod_features_map.resize(n_type);
    prod_features_map_from_keys.resize(n_type);
    linear_features.resize(n_type);

    get_types_for_feature_combinations(f_obj);

    if (separate_erased == false) {
        set_mapping_prod(f_obj, true);
    }
    else {
        set_mapping_prod(f_obj, false);
        set_mapping_prod_erased(f_obj);
    }
    set_mapping_prod_of_features(f_obj);

    set_features_using_mappings(f_obj);
    set_terms_using_mappings(f_obj, pot);

    sort_potential_model();

}

Potential::~Potential(){}

void Potential::set_mapping_prod(const Features& f_obj, const bool erased){

    std::vector<std::set<vector1i> > nonequiv_keys(n_type);

//    int count1(0), count2(0);
    const auto& mfeatures = f_obj.get_features();
    for (const auto& sfeature: mfeatures){
        const auto type1 = sfeature[0].type1;
        for (const auto& sterm: sfeature){
            for (const auto& t1: type1){
                nonequiv_keys[t1].insert(sterm.nlmtc_keys);
 //               ++count2;
            }
            if (erased == true){
                for (size_t i = 0; i < sterm.nlmtc_keys.size(); ++i){
                    //int head_key = sterm.nlmtc_keys[i];
                    const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                    for (const auto& t1: type1){
                        nonequiv_keys[t1].insert(keys);
                    }
                }
            }
        }
    }

    for (int t1 = 0; t1 < n_type; ++t1){
        nonequiv_set_to_mappings(nonequiv_keys[t1],
                                 prod_map_from_keys[t1],
                                 prod_map[t1]);
    }
}

void Potential::set_mapping_prod_erased(const Features& f_obj){

    std::vector<std::set<vector1i> > nonequiv_keys(n_type);

    const auto& mfeatures = f_obj.get_features();
    for (const auto& sfeature: mfeatures){
        const auto type1 = sfeature[0].type1;
        for (const auto& sterm: sfeature){
            for (size_t i = 0; i < sterm.nlmtc_keys.size(); ++i){
                int head_key = sterm.nlmtc_keys[i];
                bool append = true;
                if (eliminate_conj == true and
                    nlmtc_map[head_key].lm.conj == true) append = false;
                if (append == true){
                    const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                    for (const auto& t1: type1){
                        nonequiv_keys[t1].insert(keys);
                    }
                }
            }
        }
    }

    for (int t1 = 0; t1 < n_type; ++t1){
        nonequiv_set_to_mappings(nonequiv_keys[t1],
                                 prod_map_erased_from_keys[t1],
                                 prod_map_erased[t1]);
    }
}

void Potential::set_features_using_mappings(const Features& f_obj){

    const auto& mfeatures = f_obj.get_features();
    for (int t1 = 0; t1 < n_type; ++t1){
        linear_features[t1].resize(mfeatures.size());
    }

    int idx(0);
    for (const auto& sfeature: mfeatures){
        const auto type1 = sfeature[0].type1;
        for (const auto& t1: type1){
            std::unordered_map<int, double> sfeature_map;
            // finding nonequivalent features
            for (const auto& sterm: sfeature){
                const int prod_key = prod_map_from_keys[t1][sterm.nlmtc_keys];
                if (sfeature_map.count(prod_key) == 0){
                    sfeature_map[prod_key] = sterm.coeff;
                }
                else {
                    sfeature_map[prod_key] += sterm.coeff;
                }
            }
            // end: finding nonequivalent features

            for (const auto& sterm: sfeature_map){
                MappedSingleTerm msterm = {sterm.second, sterm.first};
                linear_features[t1][idx].emplace_back(msterm);
            }
        }
        ++idx;
    }
}

void Potential::set_mapping_prod_of_features(const Features& f_obj){

    std::vector<std::set<vector1i> > nonequiv_keys(n_type);

    const auto& feature_combinations = f_obj.get_feature_combinations();
    int count(0);
    for (const auto& comb: feature_combinations){
        const vector1i& type1 = type1_feature_combs[count];
        for (size_t ci = 0; ci < comb.size(); ++ci){
            vector1i keys = erase_a_key(comb, ci);
            for (const auto& t1: type1){
                nonequiv_keys[t1].insert(keys);
            }
        }
        ++count;
    }

    for (int t1 = 0; t1 < n_type; ++t1){
        nonequiv_set_to_mappings(nonequiv_keys[t1],
                                 prod_features_map_from_keys[t1],
                                 prod_features_map[t1]);
    }
}

void Potential::get_types_for_feature_combinations(const Features& f_obj){

    const auto& feature_combinations = f_obj.get_feature_combinations();
    const auto& mfeatures = f_obj.get_features();

    // finding atom types with nonzero features and feature products
    for (const auto& comb: feature_combinations){
        std::set<int> type1_intersection;
        for (size_t ci = 0; ci < comb.size(); ++ci){
            const auto& sfeature = mfeatures[comb[ci]];
            std::set<int> type1_s(sfeature[0].type1.begin(),
                                  sfeature[0].type1.end());
            if (ci == 0) type1_intersection = type1_s;
            else {
                std::set<int> result;
                std::set_intersection(type1_intersection.begin(),
                                      type1_intersection.end(),
                                      type1_s.begin(), type1_s.end(),
                                      std::inserter(result, result.end()));
                type1_intersection = result;
            }
        }
        vector1i type1(type1_intersection.begin(), type1_intersection.end());
        type1_feature_combs.emplace_back(type1);
    }
}

void Potential::set_terms_using_mappings(const Features& f_obj,
                                         const vector1d& pot){

    const auto& mfeatures = f_obj.get_features();
    const auto& feature_combinations = f_obj.get_feature_combinations();

    // finding nonequivalent potential terms
    std::vector<std::unordered_map<vector1i, vector1d, HashVI> > nonequiv_map;
    nonequiv_map.resize(n_type);

    int idx = 0;
    for (const auto& comb: feature_combinations){
        int n_prods(0);
        for (size_t ci = 0; ci < comb.size(); ++ci){
            n_prods += mfeatures[comb[ci]][0].nlmtc_keys.size();
        }

        const auto& type1 = type1_feature_combs[idx];
        for (const auto& t1: type1){
            for (size_t ci = 0; ci < comb.size(); ++ci){
                int head_c = comb[ci];
                vector1i f_keys = erase_a_key(comb, ci);
                const int prod_features_key
                        = prod_features_map_from_keys[t1][f_keys];
                const auto& sfeature = mfeatures[head_c];
                for (const auto& sterm: sfeature){
                    const int n_order = sterm.nlmtc_keys.size();
                    const double coeff_f = pot[idx] * sterm.coeff;
                    const double coeff_e = coeff_f / double(n_prods);
                    for (int i = 0; i < n_order; ++i){
                        const int head_key = sterm.nlmtc_keys[i];
                        vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                        int prod_key;
                        if (separate_erased == true){
                            prod_key = prod_map_erased_from_keys[t1][keys];
                        }
                        else {
                            prod_key = prod_map_from_keys[t1][keys];
                        }

                        vector1i keys_all = {head_key,
                                             prod_key,
                                             prod_features_key,
                                             idx};

                        bool append = true;
                        if (eliminate_conj == true and
                            nlmtc_map[head_key].lm.conj == true) append = false;

                        if (append == true){
                            if (nonequiv_map[t1].count(keys_all) == 0){
                                nonequiv_map[t1][keys_all]
                                    = vector1d{coeff_e,coeff_f};
                            }
                            else {
                                nonequiv_map[t1][keys_all][0] += coeff_e;
                                nonequiv_map[t1][keys_all][1] += coeff_f;
                            }
                        }
                    }
                }
            }
        }
        ++idx;
    }
    // end: finding nonequivalent potential terms

    for (int t1 = 0; t1 < n_type; ++t1){
        for (const auto& term: nonequiv_map[t1]){
            const double coeff_e = term.second[0];
            const double coeff_f = term.second[1];
            const int head_key = term.first[0];
            const int prod_key = term.first[1];
            const int prod_features_key = term.first[2];
            const int feature_idx = term.first[3];
            PotentialTerm pterm = {coeff_e,
                                   coeff_f,
                                   head_key,
                                   prod_key,
                                   prod_features_key,
                                   feature_idx};

            if (eliminate_conj == false){
                potential_model_each_key[t1][head_key].emplace_back(pterm);
            }
            else {
                const auto& nlmtc = nlmtc_map[head_key];
                if (nlmtc.lm.conj == false){
                    int noconj_key = nlmtc.nlmtc_noconj_key;
                    potential_model_each_key[t1][noconj_key]
                                                .emplace_back(pterm);
                }
            }
        }
    }
}

void Potential::sort_potential_model(){

    // sorted by prod_key and then by prod_features_key
    for (int t1 = 0; t1 < n_type; ++t1){
        for (auto& pmodel: potential_model_each_key[t1]){
            std::sort(pmodel.begin(), pmodel.end(),
                    [](const PotentialTerm& lhs, const PotentialTerm& rhs){
                    if (lhs.prod_key != rhs.prod_key){
                        return lhs.prod_key < rhs.prod_key;
                    }
                    else {
                        return lhs.prod_features_key < rhs.prod_features_key;
                    }
                    });
        }
    }
}


void Potential::nonequiv_set_to_mappings
(const std::set<vector1i>& nonequiv_keys,
 ProdMapFromKeys& map_from_keys,
 vector2i& map){

    map = vector2i(nonequiv_keys.begin(), nonequiv_keys.end());
    std::sort(map.begin(), map.end());

    int i(0);
    for (const auto& keys: map){
        map_from_keys[keys] = i;
        ++i;
    }
}

vector1i Potential::erase_a_key(const vector1i& original, const int idx){
    vector1i keys = original;
    keys.erase(keys.begin() + idx);
    std::sort(keys.begin(), keys.end());
    return keys;
}

void Potential::print_keys(const vector1i& keys){
    for (const auto& k: keys)
        std::cout << k << " ";
    std::cout << std::endl;
}

const std::vector<lmAttribute>& Potential::get_lm_map() const {
    return lm_map;
}
const std::vector<nlmtcAttribute>&
Potential::get_nlmtc_map_no_conjugate() const{
    return nlmtc_map_no_conjugate;
}
const std::vector<nlmtcAttribute>& Potential::get_nlmtc_map() const {
    return nlmtc_map;
}
const std::vector<ntcAttribute>& Potential::get_ntc_map() const {
    return ntc_map;
}

const vector2i& Potential::get_prod_map(const int t) const {
    return prod_map[t];
}
const vector2i& Potential::get_prod_map_erased(const int t) const {
    return prod_map_erased[t];
}
const vector2i& Potential::get_prod_features_map(const int t) const {
    return prod_features_map[t];
}
const int Potential::get_n_nlmtc_all() const {
    return n_nlmtc_all;
}
/*
const vector2i& Potential::get_prod_map_type() const {
    return prod_map_type;
}
const vector2i& Potential::get_prod_map_erased_type() const {
    return prod_map_erased_type;
}
*/

const MappedMultipleFeatures&
Potential::get_linear_features(const int t) const {
    return linear_features[t];
}

const PotentialModel& Potential::get_potential_model(const int type1,
                                                     const int head_key) const {
    return potential_model_each_key[type1][head_key];
}
