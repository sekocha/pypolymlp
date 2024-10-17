/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_mapping.h"

Mapping::Mapping(){}
Mapping::Mapping(const struct feature_params& fp){

    n_type = fp.n_type;
    n_fn = fp.params.size();
    maxl = fp.maxl;

    set_type_pairs(fp);
    set_map_n_to_tplist();
    if (fp.feature_type == "pair") set_ntp_attrs();
    else if (fp.feature_type == "gtinv") set_nlmtp_attrs();
}

Mapping::~Mapping(){}

void Mapping::set_type_pairs(const struct feature_params& fp){

    type_pairs.resize(n_type);
    for (int i = 0; i < n_type; ++i) type_pairs[i].resize(n_type);

    int tp = 0;
    for (int i = 0; i < n_type; ++i){
        for (int j = 0; j < n_type; ++j){
            if (i <= j){
                type_pairs[i][j] = type_pairs[j][i] = tp;
                map_tp_to_nlist.emplace_back(fp.params_conditional[i][j]);
                vector2d params_match;
                for (const auto& n: map_tp_to_nlist[tp]){
                    params_match.emplace_back(fp.params[n]);
                }
                map_tp_to_params.emplace_back(params_match);
                ++tp;
            }
        }
    }
    n_type_pairs = tp;
}

void Mapping::set_type_pairs_charge(const feature_params& fp){

    int tp = 0;
    type_pairs.resize(n_type);
    for (int i = 0; i < n_type; ++i){
        for (int j = 0; j < n_type; ++j){
            type_pairs[i].emplace_back(tp);
            map_tp_to_nlist.emplace_back(fp.params_conditional[i][j]);
            ++tp;
        }
    }
    n_type_pairs = n_type * n_type;
}

void Mapping::set_map_n_to_tplist(){

    map_n_to_tplist.resize(n_fn);
    n_id_list.resize(n_type_pairs);
    for (int tp = 0; tp < n_type_pairs; ++tp){
        n_id_list[tp].resize(n_fn);
        int id = 0;
        for (const int n: map_tp_to_nlist[tp]){
            map_n_to_tplist[n].emplace_back(tp);
            n_id_list[tp][n] = id;
            ++id;
        }
    }
}

void Mapping::set_ntp_attrs(){

    int ntp_key(0), n_id;
    for (int tp = 0; tp < n_type_pairs; ++tp){
        for (const auto& n: map_tp_to_nlist[tp]){
            n_id = n_id_list[tp][n];
            ntpAttr ntp = {n, n_id, tp, ntp_key};
            ntp_attrs.emplace_back(ntp);
            map_ntp_to_key[vector1i({n, tp})] = ntp_key;
            ++ntp_key;
        }
    }
    n_ntp_all = ntp_attrs.size();

    ntp_attrs_type.resize(n_type);
    for (const auto& ntp: ntp_attrs){
        for (int type1 = 0; type1 < n_type; ++type1){
            const auto& tp_array = type_pairs[type1];
            if (std::find(tp_array.begin(), tp_array.end(), ntp.tp) != tp_array.end()){
                ntp_attrs_type[type1].emplace_back(ntp);
            }
        }
    }
}

void Mapping::set_nlmtp_attrs(){

    set_lm_attrs();
    int nlmtp_key(0), nlmtp_noconj_key(0), conj_key, conj_key_add, n_id;
    for (int n = 0; n < n_fn; ++n){
        const auto& tp_list = map_n_to_tplist[n];
        for (int lm = 0; lm < n_lm_all; ++lm){
            const auto& lm_attr = lm_attrs[lm];
            conj_key_add = 2 * lm_attr.m * tp_list.size();
            for (const auto& tp: tp_list){
                conj_key = nlmtp_key - conj_key_add;
                n_id = n_id_list[tp][n];
                nlmtpAttr nlmtps = {
                    n, n_id, lm_attr, tp, nlmtp_key, conj_key, nlmtp_noconj_key
                };
                nlmtp_attrs.emplace_back(nlmtps);
                map_nlmtp_to_key[vector1i({n, lm, tp})] = nlmtp_key;
                ++nlmtp_key;
                if (lm_attr.conj == false) {
                    nlmtp_attrs_no_conjugate.emplace_back(nlmtps);
                    ++nlmtp_noconj_key;
                }
            }
        }
    }
    n_nlmtp_all = nlmtp_attrs.size();
}


void Mapping::set_lm_attrs(){

    int ylm_key;
    double cc, sign_j;
    bool conj;
    for (int l = 0; l < maxl + 1; ++l){
        if (l % 2 == 0){
            sign_j = 1;
        }
        else {
            sign_j = -1;
        }
        for (int m = -l; m < l + 1; ++m){
            if (m % 2 == 0) {
                cc = 1;
            }
            else {
                cc = -1;
            }
            if (m < 1) {
                ylm_key = (l+3)*l/2 + m;
                conj = false;
            }
            else {
                ylm_key = (l+3)*l/2 - m;
                conj = true;
            }
            lmAttr lm_attr = {l, m, ylm_key, conj, cc, sign_j};
            lm_attrs.emplace_back(lm_attr);
        }
    }
    n_lm_all = lm_attrs.size();
    n_lm = (n_lm_all + maxl + 1) / 2;
}


const int Mapping::get_n_type_pairs() const { return n_type_pairs; }
const int Mapping::get_n_ntp_all() const { return n_ntp_all; }
const int Mapping::get_n_nlmtp_all() const { return n_nlmtp_all; }

const vector2i& Mapping::get_type_pairs() const {
    return type_pairs;
}
const vector2i& Mapping::get_type_pair_to_nlist() const {
    return map_tp_to_nlist;
}
const vector2i& Mapping::get_n_to_type_pairs() const {
    return map_n_to_tplist;
}
const vector2i& Mapping::get_n_ids() const {
    return n_id_list;
}
const vector3d& Mapping::get_type_pair_to_params() const {
    return map_tp_to_params;
}

const std::vector<ntpAttr>& Mapping::get_ntp_attrs() const {
    return ntp_attrs;
}
const std::vector<ntpAttr>& Mapping::get_ntp_attrs(const int type1) const {
    return ntp_attrs_type[type1];
}
const std::vector<nlmtpAttr>& Mapping::get_nlmtp_attrs_no_conjugate() const {
    return nlmtp_attrs_no_conjugate;
}
const std::vector<nlmtpAttr>& Mapping::get_nlmtp_attrs() const {
    return nlmtp_attrs;
}
const std::vector<lmAttr>& Mapping::get_lm_attrs() const {
    return lm_attrs;
}

const MapFromVec& Mapping::get_ntp_to_key() const {
    return map_ntp_to_key;
}
const MapFromVec& Mapping::get_nlmtp_to_key() const {
    return map_nlmtp_to_key;
}
