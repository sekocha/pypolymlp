/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
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


#ifndef __POLYMLP_STRUCTS
#define __POLYMLP_STRUCTS

#include "polymlp_mlpcpp.h"


struct ntpAttr {
    int n;
    int n_id;
    int tp;
    int global_id;

    int ilocal_id;
    int jlocal_id;
};

struct lmAttr {
    int l;
    int m;
    int ylmkey;
    bool conj;
    double cc_coeff;
    double sign_j;
};

struct nlmtpAttr {
    int n;
    int n_id;
    lmAttr lm;
    int tp;

    int global_id;
    int global_noconj_id;
    int global_conj_id;

    int ilocal_id;
    int ilocal_noconj_id;
    int jlocal_noconj_id;
    int ilocal_conj_id;
};


typedef std::unordered_map<vector1i,int,HashVI> MapFromVec;
typedef std::vector<ntpAttr> ntpAttrArray;
typedef std::vector<nlmtpAttr> nlmtpAttrArray;
typedef std::vector<lmAttr> lmAttrArray;

struct LinearTerm {
    int n;
    int lm_comb_id;
    int tp_comb_id;
    int order;
    vector1i type1;
};

struct SingleTerm {
    double coeff;
    vector1i nlmtp_ids;
};

struct MappedSingleTerm {
    double coeff;
    int id;
};

struct MappedSingleDerivTerm {
    double coeff;
    int prod_id;
    int head_id;
    int feature_id;
};


typedef std::vector<SingleTerm> SingleFeature;
typedef std::vector<SingleFeature> MultipleFeatures;
typedef std::vector<MappedSingleTerm> MappedSingleFeature;
typedef std::vector<MappedSingleFeature> MappedMultipleFeatures;
typedef std::vector<MappedSingleDerivTerm> MappedSingleFeatureDeriv;
typedef std::vector<MappedSingleFeatureDeriv> MappedMultipleFeaturesDeriv;


struct PolynomialTerm {
    int global_id;
    vector1i local_ids;
};

struct PotentialTerm {
    double coeff_e;
    double coeff_f;
    int prod_id;
    int prod_features_id;
};

typedef std::vector<PotentialTerm> PotentialModel;


struct MapsType {
    ntpAttrArray ntp_attrs;
    nlmtpAttrArray nlmtp_attrs;
    nlmtpAttrArray nlmtp_attrs_noconj;

    MultipleFeatures features;
    std::vector<PolynomialTerm> polynomial;

    bool is_conj(const int local_id){
        return nlmtp_attrs[local_id].lm.conj;
    }

    int get_noconj_id(const int local_id){
        return nlmtp_attrs[local_id].ilocal_noconj_id;
    }

    int get_feature_size(const int local_feature_id){
        return features[local_feature_id][0].nlmtp_ids.size();
    }

    void clear(){
        features = {};
    }
};


struct Maps {
    ntpAttrArray ntp_attrs;
    nlmtpAttrArray nlmtp_attrs;
    nlmtpAttrArray nlmtp_attrs_noconj;
    lmAttrArray lm_attrs;

    MapFromVec ntp_to_global;      // (n, tp) --> ntp_global key
    MapFromVec ntp_global_to_iloc; // (ntp_global, type1) -> ntp_local

    MapFromVec nlmtp_to_global;      // (n, lm, tp) --> nlmtp_global key
    MapFromVec nlmtp_global_to_iloc; // (nlmtp_global, type1) -> nlmtp_local
    vector1i global_to_global_conj;  // nlmtp_global -> nlmtp_global_conj

    vector2i type_pairs;             // (type1, type2) --> type_pairs
    vector2i tp_to_types;            // type_pair --> (type1, type2)

    vector2i tp_to_n;                // type_pair --> conditional n values
    vector3d tp_to_params;           // type_pair -> params
    vector2i n_to_tp;                // n --> type_pairs
    vector2i tpn_to_n_id;            // (type_pair, n) -> n_id for type_pair

    std::vector<MapsType> maps_type;

    int get_n_type(){ return type_pairs.size(); }
    int get_n_type_pairs(){ return tp_to_types.size(); }
    int get_n_lm(){ return lm_attrs.size(); }

    int get_feature_size(const int type1, const int local_prod_id){
        return maps_type[type1].get_feature_size(local_prod_id);
    }

    vector1i nlmtp_vec_to_global(const int n, const vector1i& lms, const vector1i& tps){
        vector1i global_keys;
        for (size_t i = 0; i < lms.size(); ++i){
            global_keys.emplace_back(nlmtp_to_global[vector1i({n, lms[i], tps[i]})]);
        }
        return global_keys;
    }

    vector1i nlmtp_global_vec_to_iloc(const vector1i& global_keys, const int& type1){
        vector1i local_keys;
        for (const int global: global_keys){
            vector1i key = {global, type1};
            local_keys.emplace_back(nlmtp_global_to_iloc[key]);
        }
        return local_keys;
    }

    void clear(){
        for (auto& mt1: maps_type) mt1.clear();
        ntp_to_global = {};
        ntp_global_to_iloc = {};
        nlmtp_to_global = {};
        nlmtp_global_to_iloc = {};
        global_to_global_conj = {};
    }
};


inline void convert_set_to_mappings(
    const std::set<vector1i>& uniq_keys,
    MapFromVec& map_from_keys,
    vector2i& map
){

    map = vector2i(uniq_keys.begin(), uniq_keys.end());
    std::sort(map.begin(), map.end());

    int i(0);
    for (const auto& keys: map){
        map_from_keys[keys] = i;
        ++i;
    }
}

inline vector1i erase_a_key(const vector1i& original, const int idx){
    vector1i keys = original;
    keys.erase(keys.begin() + idx);
    std::sort(keys.begin(), keys.end());
    return keys;
}

#endif
