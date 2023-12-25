/**************************************************************************** 

        Copyright (C) 2021 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to
        the Free Software Foundation, Inc., 51 Franklin Street,
        Fifth Floor, Boston, MA 02110-1301, USA, or see
        http://www.gnu.org/copyleft/gpl.txt

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
    potential_model_each_key = PotentialModelEachKey(size);

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
//    set_potential_model_each_prod_key();

}

Potential::~Potential(){}

void Potential::set_mapping_prod(const Features& f_obj, const bool erased){

    std::set<vector1i> nonequiv_keys;
    const auto& mfeatures = f_obj.get_features();
    for (const auto& sfeature: mfeatures){
        for (const auto& sterm: sfeature){
            nonequiv_keys.insert(sterm.nlmtc_keys);
            if (erased == true){
                for (int i = 0; i < sterm.nlmtc_keys.size(); ++i){
                    int head_key = sterm.nlmtc_keys[i];
                    const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                    nonequiv_keys.insert(keys);
                }
            }
        }
    }
    nonequiv_set_to_mappings(nonequiv_keys, prod_map_from_keys, prod_map);
}

void Potential::set_mapping_prod_erased(const Features& f_obj){

    std::set<vector1i> nonequiv_keys;
    const auto& mfeatures = f_obj.get_features();
    for (const auto& sfeature: mfeatures){
        for (const auto& sterm: sfeature){
            for (int i = 0; i < sterm.nlmtc_keys.size(); ++i){
                int head_key = sterm.nlmtc_keys[i];
                bool append = true;
                if (eliminate_conj == true and 
                    nlmtc_map[head_key].lm.conj == true) append = false;
                if (append == true){
                    const vector1i keys = erase_a_key(sterm.nlmtc_keys, i);
                    nonequiv_keys.insert(keys);
                }
            }
        }
    }
    nonequiv_set_to_mappings(nonequiv_keys, 
                             prod_map_erased_from_keys, 
                             prod_map_erased);
}

void Potential::set_features_using_mappings(const Features& f_obj){

    const auto& mfeatures = f_obj.get_features();
    linear_features.resize(mfeatures.size());
    int idx(0);
    for (const auto& sfeature: mfeatures){
        // finding nonequivalent features
        std::unordered_map<int, double> sfeature_map;
        for (const auto& sterm: sfeature){
            const int prod_key = prod_map_from_keys[sterm.nlmtc_keys];
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
            linear_features[idx].emplace_back(msterm);
        }
        ++idx;
    }
}

void Potential::set_mapping_prod_of_features(const Features& f_obj){

    const auto& feature_combinations = f_obj.get_feature_combinations();

    std::set<vector1i> nonequiv_keys;
    for (const auto& comb: feature_combinations){
        for (int ci = 0; ci < comb.size(); ++ci){
            vector1i keys = erase_a_key(comb, ci);
            nonequiv_keys.insert(keys);
        }
    }
    nonequiv_set_to_mappings(nonequiv_keys, 
                             prod_features_map_from_keys, 
                             prod_features_map);

}

void Potential::set_terms_using_mappings(const Features& f_obj, 
                                         const vector1d& pot){

    const auto& mfeatures = f_obj.get_features();
    const auto& feature_combinations = f_obj.get_feature_combinations();

    // finding nonequivalent potential terms
    std::unordered_map<vector1i, vector1d, HashVI> nonequiv_map;
    int idx = 0;
    for (const auto& comb: feature_combinations){
        int n_prods(0);
        for (int ci = 0; ci < comb.size(); ++ci){
            n_prods += mfeatures[comb[ci]][0].nlmtc_keys.size();
        }
        for (int ci = 0; ci < comb.size(); ++ci){
            int head_c = comb[ci];
            vector1i f_keys = erase_a_key(comb, ci);
            const int prod_features_key = prod_features_map_from_keys[f_keys];
            const auto& sfeature = mfeatures[head_c];
            for (const auto& sterm: sfeature){
                const int n_order = sterm.nlmtc_keys.size();
                const double coeff_f = pot[idx] * sterm.coeff;
                const double coeff_e = coeff_f / double(n_prods);
                for (int i = 0; i < n_order; ++i){
                    const int head_key = sterm.nlmtc_keys[i];
                    vector1i keys  = erase_a_key(sterm.nlmtc_keys, i);
                    int prod_key;
                    if (separate_erased == true){
                        prod_key = prod_map_erased_from_keys[keys];
                    }
                    else {
                        prod_key = prod_map_from_keys[keys];
                    }
                    vector1i keys_all = {head_key, 
                                         prod_key, 
                                         prod_features_key,
                                         idx};

                    bool append = true;
                    if (eliminate_conj == true and 
                        nlmtc_map[head_key].lm.conj == true) append = false;

                    if (append == true){
                        if (nonequiv_map.count(keys_all) == 0){
                            nonequiv_map[keys_all] = vector1d{coeff_e,coeff_f};
                        }
                        else {
                            nonequiv_map[keys_all][0] += coeff_e;
                            nonequiv_map[keys_all][1] += coeff_f;
                        }
                    }
                }
            }
        }
        ++idx;
    }
    // end: finding nonequivalent potential terms

    for (const auto& term: nonequiv_map){
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
            potential_model_each_key[head_key].emplace_back(pterm);
        }
        else {
            const auto& nlmtc = nlmtc_map[head_key];
            if (nlmtc.lm.conj == false){
                int noconj_key = nlmtc.nlmtc_noconj_key;
                potential_model_each_key[noconj_key].emplace_back(pterm);
            }
        }
    }
}

void Potential::sort_potential_model(){

    // sorted by prod_key and then by prod_features_key
    for (auto& pmodel: potential_model_each_key){
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

/*
void Potential::set_potential_model_each_prod_key(){

    // sorted by prod_key and then by prod_features_key
    int n_prod_keys;
    if (separate_erased == true) n_prod_keys = prod_map_erased.size();
    else {
        exit(8);
    }

    potential_model_each_head_and_prod_key.resize(n_prod_keys);
    for (int i = 0; i < n_prod_keys; ++i){
        potential_model_each_head_and_prod_key[i].resize
                                    (potential_model_each_key.size());
    }

    int head_key(0);
    for (auto& pmodel: potential_model_each_key){
        for (auto& pterm: pmodel){
            potential_model_each_head_and_prod_key
                [pterm.prod_key][head_key].emplace_back(pterm);
        }
        ++head_key;
    }
}
*/

void Potential::nonequiv_set_to_mappings
(const std::set<vector1i>& nonequiv_keys,
 std::unordered_map<vector1i, int, HashVI>& map_from_keys,
 vector2i& map){

    int i(0);
    map = vector2i(nonequiv_keys.size());
    for (const auto& keys: nonequiv_keys){
        map_from_keys[keys] = i;
        map[i] = keys;
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

const vector2i& Potential::get_prod_map() const { 
    return prod_map; 
}
const vector2i& Potential::get_prod_map_erased() const { 
    return prod_map_erased; 
}
const vector2i& Potential::get_prod_features_map() const { 
    return prod_features_map; 
}
const int Potential::get_n_nlmtc_all() const { 
    return n_nlmtc_all; 
}

const MappedMultipleFeatures& Potential::get_linear_features() const { 
    return linear_features; 
}
const PotentialModel& Potential::get_potential_model() const { 
    return potential_model; 
}
const PotentialModel& Potential::get_potential_model(const int head_key) const {
    return potential_model_each_key[head_key]; 
}
//const PotentialModel& Potential::get_potential_model(const int head_key, 
//                                                     const int prod_key) const {
//    return potential_model_each_head_and_prod_key[prod_key][head_key]; 
//}

