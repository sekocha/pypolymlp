/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        SingleTerm: [coeff,[(n1,l1,m1,tc1), (n2,l2,m2,tc2), ...]]
            (n1,l1,m1,tc1) is represented with nlmtc_key.

        SingleFeature: [SingleTerms1, SingleTerms2, ...]

        MultipleFeatures: [SingleFeature1, SingleFeature2, ...]

*****************************************************************************/


#include "polymlp_features_utils.h"


int set_linear_features_pair(Maps& maps){

    int type1, type2, local_id;
    vector1i key;
    for (const auto& ntp: maps.ntp_attrs){
        const auto& types = maps.tp_to_types[ntp.tp];
        type1 = types[0];
        type2 = types[1];
        key = {ntp.global_id, type1};
        local_id = maps.ntp_global_to_iloc[key];
        SingleTerm single = {1.0, vector1i({local_id})};
        SingleFeature feature_local = {single};
        maps.maps_type[type1].features.emplace_back(feature_local);
        if (type1 != type2){
            key = {ntp.global_id, type2};
            local_id = maps.ntp_global_to_iloc[key];
            single = {1.0, vector1i({local_id})};
            feature_local = {single};
            maps.maps_type[type2].features.emplace_back(feature_local);
        }
    }
    return 0;
}

int set_linear_features_gtinv(
    const feature_params& fp,
    const ModelParams& modelp,
    Maps& maps
){
    const vector3i& lm_array = fp.lm_array;
    const vector2d& lm_coeffs = fp.lm_coeffs;
    const auto& tp_combs = modelp.get_tp_combs();

    for (const auto& linear: modelp.get_linear_terms()){
        const auto& tp_comb = tp_combs[linear.order][linear.tp_comb_id];
        const auto& lm_list = lm_array[linear.lm_comb_id];
        const auto& coeff_list = lm_coeffs[linear.lm_comb_id];
        for (const int t: linear.type1){
            SingleFeature feature_local;
            for (size_t i = 0; i < lm_list.size(); ++i){
                auto global_ids = maps.nlmtp_vec_to_global(
                    linear.n, lm_list[i], tp_comb
                );
                std::sort(global_ids.begin(), global_ids.end());
                const auto local_ids = maps.nlmtp_global_vec_to_iloc(global_ids, t);
                SingleTerm single = {coeff_list[i], local_ids};
                feature_local.emplace_back(single);
            }
            maps.maps_type[t].features.emplace_back(feature_local);
        }
    }
    return 0;
}


int get_linear_features_gtinv_with_reps(
    const feature_params& fp,
    const ModelParams& modelp,
    Maps& maps,
    std::vector<MultipleFeatures> features_for_map
){

    const vector3i& lm_array = fp.lm_array;
    const vector2d& lm_coeffs = fp.lm_coeffs;
    const auto& tp_combs = modelp.get_tp_combs();

    features_for_map.resize(n_type);
    for (const auto& linear: modelp.get_linear_terms()){
        const auto& tp_comb = tp_combs[linear.order][linear.tp_comb_id];
        const auto& lm_list = lm_array[linear.lm_comb_id];
        const auto& coeff_list = lm_coeffs[linear.lm_comb_id];
        for (const int t: linear.type1){
            SingleFeature feature_local;
            for (size_t i = 0; i < lm_list.size(); ++i){
                vector1i local_ids;
                find_local_ids(maps, t, linear.n, lm_list[i], tp_comb, local_ids);
                SingleTerm single = {coeff_list[i], local_ids};
                feature_local.emplace_back(single);
            }
            features_for_map[t].emplace_back(feature_local);
        }
    }
    return 0;
}


int find_local_ids(
    Maps& maps,
    const int type1,
    const int n,
    const vector1i& lm_comb,
    const vector1i& tp_comb,
    vector1i& local_ids
){
    auto global_ids1 = maps.nlmtp_vec_to_global(n, lm_comb, tp_comb);
    std::sort(global_ids1.begin(), global_ids1.end());

    vector1i global_ids2;
    for (const auto& id: global_ids1)
        global_ids2.emplace_back(maps.global_to_global_conj[id]);
    std::sort(global_ids2.begin(), global_ids2.end());

    std::set<vector1i> sort;
    sort.insert(global_ids1);
    sort.insert(global_ids2);
    const auto& rep = *sort.begin();
    local_ids = maps.nlmtp_global_vec_to_iloc(rep, type1);

    return 0;
}


int get_nonequiv_ids(const MultipleFeatures& features, std::set<vector1i> nonequiv){

    for (const auto& sfeature: features){
        for (const auto& sterm: sfeature){
            nonequiv.insert(sterm.nlmtp_ids);
        }
    }
    return 0;
}

int get_nonequiv_deriv_ids(
    const MultipleFeatures& features,
    MapsType& maps_type,
    const bool eliminate_conj,
    std::set<vector1i> nonequiv
){
    for (const auto& sfeature: features){
        for (const auto& sterm: sfeature){
            for (size_t i = 0; i < sterm.nlmtp_ids.size(); ++i){
                const int head_id = sterm.nlmtp_ids[i];
                if (eliminate_conj == false or maps_type.is_conj(head_id) == false)
                     nonequiv.insert(erase_a_key(sterm.nlmtp_ids, i));
            }
        }
    }
    return 0;
}

int convert_to_mapped_features_algo1(
    const MultipleFeatures& features,
    MapFromVec& prod_map_from_keys,
    const vector2i& prod,
    MappedMultipleFeatures& mapped_features
){
    mapped_features.resize(prod.size());
    for (int f_id = 0; f_id < features.size(); ++f_id){
        std::unordered_map<int, double> sfeature_map;
        for (const auto& sterm: features[f_id]){
            const int prod_id = prod_map_from_keys[sterm.nlmtp_ids];
            if (sfeature_map.count(prod_id) == 0)
                sfeature_map[prod_id] = sterm.coeff;
            else sfeature_map[prod_id] += sterm.coeff;
        }
        for (const auto& sterm: sfeature_map){
            MappedSingleTerm msterm = {sterm.second, f_id};
            mapped_features[sterm.first].emplace_back(msterm);
        }
    }
    return 0;
}

int convert_to_mapped_features_algo2(
    const MultipleFeatures& features,
    MapFromVec& prod_map_from_keys,
    MappedMultipleFeatures& mapped_features
){
    mapped_features.resize(features.size());
    for (int f_id = 0; f_id < features.size(); ++f_id){
        std::unordered_map<int, double> sfeature_map;
        for (const auto& sterm: features[f_id]){
            const int prod_id = prod_map_from_keys[sterm.nlmtp_ids];
            if (sfeature_map.count(prod_id) == 0)
                sfeature_map[prod_id] = sterm.coeff;
            else sfeature_map[prod_id] += sterm.coeff;
        }
        for (const auto& sterm: sfeature_map){
            MappedSingleTerm msterm = {sterm.second, sterm.first};
            mapped_features[f_id].emplace_back(msterm);
        }
    }
    return 0;
}
