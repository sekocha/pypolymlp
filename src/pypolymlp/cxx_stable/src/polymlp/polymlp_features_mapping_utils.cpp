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


#include "polymlp_features_mapping_utils.h"


int get_nonequiv_ids(const MultipleFeatures& features, std::set<vector1i>& nonequiv){

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
    std::set<vector1i>& nonequiv
){
    // TODO: Interface for eliminate_conj and is_conj.
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


int _convert_single_feature_to_map(
    const SingleFeature& sfeature,
    MapFromVec& prod_map_from_keys,
    std::unordered_map<int, double>& sfeature_map
){
    for (const auto& sterm: sfeature){
        const int prod_id = prod_map_from_keys[sterm.nlmtp_ids];
        if (sfeature_map.count(prod_id) == 0)
            sfeature_map[prod_id] = sterm.coeff;
        else sfeature_map[prod_id] += sterm.coeff;
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
        std::unordered_map<int, double> smap;
        _convert_single_feature_to_map(features[f_id], prod_map_from_keys, smap);
        for (const auto& sterm: smap){
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
        std::unordered_map<int, double> smap;
        _convert_single_feature_to_map(features[f_id], prod_map_from_keys, smap);
        for (const auto& sterm: smap){
            MappedSingleTerm msterm = {sterm.second, sterm.first};
            mapped_features[f_id].emplace_back(msterm);
        }
    }
    return 0;
}

int convert_to_mapped_features_deriv_algo1(
    const MultipleFeatures& features,
    MapFromVec& prod_map_deriv_from_keys,
    const vector2i& prod_deriv,
    MappedMultipleFeaturesDeriv& mapped_features_deriv
){
    mapped_features_deriv.resize(prod_deriv.size());
    for (int f_id = 0; f_id < features.size(); ++f_id){
        std::unordered_map<vector1i, double, HashVI> smap;
        for (const auto& sterm: features[f_id]){
            for (size_t i = 0; i < sterm.nlmtp_ids.size(); ++i){
                const vector1i keys = erase_a_key(sterm.nlmtp_ids, i);
                const int prod_id = prod_map_deriv_from_keys[keys];
                const int head_id = sterm.nlmtp_ids[i];
                vector1i map_key = {head_id, prod_id};
                if (smap.count(map_key) == 0)
                    smap[map_key] = sterm.coeff;
                else smap[map_key] += sterm.coeff;
            }
        }
        for (const auto& sterm: smap){
            MappedSingleDerivTerm msterm = {sterm.second, sterm.first[0], f_id};
            mapped_features_deriv[sterm.first[1]].emplace_back(msterm);
        }
    }
    return 0;
}


int convert_to_mapped_features_deriv_algo2(
    const MultipleFeatures& features,
    MapFromVec& prod_map_deriv_from_keys,
    MappedMultipleFeaturesDeriv& mapped_features_deriv
){
    mapped_features_deriv.resize(features.size());
    for (int f_id = 0; f_id < features.size(); ++f_id){
        std::unordered_map<vector1i, double, HashVI> smap;
        for (const auto& sterm: features[f_id]){
            for (size_t i = 0; i < sterm.nlmtp_ids.size(); ++i){
                const vector1i keys = erase_a_key(sterm.nlmtp_ids, i);
                const int prod_id = prod_map_deriv_from_keys[keys];
                const int head_id = sterm.nlmtp_ids[i];
                vector1i map_key = {head_id, prod_id};
                if (smap.count(map_key) == 0)
                    smap[map_key] = sterm.coeff;
                else smap[map_key] += sterm.coeff;
            }
        }
        for (const auto& sterm: smap){
            MappedSingleDerivTerm msterm = {sterm.second, sterm.first[1], sterm.first[0]};
            mapped_features_deriv[f_id].emplace_back(msterm);
        }
    }
    sort(mapped_features_deriv);
    return 0;
}

void sort(MappedMultipleFeaturesDeriv& mapped_features_deriv){
    for (auto& mf: mapped_features_deriv){
        std::sort(mf.begin(), mf.end(), [](
            const MappedSingleDerivTerm& lhs, const MappedSingleDerivTerm& rhs
            ){
            if (lhs.head_id != rhs.head_id) return lhs.head_id < rhs.head_id;
            else return lhs.prod_id < rhs.prod_id;
        });
    }
}
