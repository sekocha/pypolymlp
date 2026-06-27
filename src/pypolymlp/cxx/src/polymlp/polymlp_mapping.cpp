/****************************************************************************
        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_mapping.h"


Mapping::Mapping(){}
Mapping::Mapping(const struct feature_params& fp){

    n_type = fp.n_type;
    n_fn = fp.params.size();
    maxl = fp.maxl;

    maps.maps_type.resize(n_type);
    set_type_pairs(fp);
    set_map_n_to_tplist();
    if (fp.feature_type == "pair"){
        set_ntp_global_attrs();
        set_ntp_local_attrs();
        set_ntp_local_attrs_compact();
    }
    else if (fp.feature_type == "gtinv"){
        set_lm_attrs();
        set_nlmtp_global_attrs();
        set_nlmtp_local_attrs();
        set_nlmtp_local_conj_ids();
        set_nlmtp_local_attrs_compact();
    }
}

Mapping::~Mapping(){}

void Mapping::set_type_pairs(const struct feature_params& fp){

    maps.type_pairs = vector2i(n_type, vector1i(n_type, 0));

    int tp = 0;
    for (int i = 0; i < n_type; ++i)
    for (int j = i; j < n_type; ++j){
        maps.type_pairs[i][j] = maps.type_pairs[j][i] = tp;
        maps.tp_to_types.emplace_back(vector1i({i, j}));

        const auto& target_conditional = fp.params_conditional[i][j];
        maps.tp_to_n.emplace_back(target_conditional);

        vector2d params_match;
        for (const auto& n: target_conditional){
            params_match.emplace_back(fp.params[n]);
        }
        maps.tp_to_params.emplace_back(params_match);
        ++tp;
    }
}

void Mapping::set_map_n_to_tplist(){

    const int n_type_pairs = maps.get_n_type_pairs();
    n_to_tp.resize(n_fn);
    tpn_to_n_id.resize(n_type_pairs);
    for (int tp = 0; tp < n_type_pairs; ++tp){
        tpn_to_n_id[tp].resize(n_fn);
        int id = 0;
        for (const int n: maps.tp_to_n[tp]){
            n_to_tp[n].emplace_back(tp);
            tpn_to_n_id[tp][n] = id;
            ++id;
        }
    }
}


void Mapping::set_ntp_global_attrs(){

    const int n_type_pairs = maps.get_n_type_pairs();
    int global_id(0);
    for (int tp = 0; tp < n_type_pairs; ++tp){
        for (const auto& n: maps.tp_to_n[tp]){
            const int n_id = tpn_to_n_id[tp][n];
            ntpAttr ntp = {n, n_id, tp, global_id};
            maps.ntp_attrs.emplace_back(ntp);
            maps.ntp_to_global[vector1i({n, tp})] = global_id;
            ++global_id;
        }
    }
}

void Mapping::set_ntp_local_attrs(){

    auto& maps_type = maps.maps_type;
    for (const auto& ntp: maps.ntp_attrs){
        const int tp = ntp.tp;
        const int type1 = maps.tp_to_types[tp][0];
        const int type2 = maps.tp_to_types[tp][1];

        const int global_id = ntp.global_id;
        const int local_id1 = maps_type[type1].ntp_attrs.size();
        const int local_id2 = maps_type[type2].ntp_attrs.size();

        auto ntp_type1 = ntp;
        ntp_type1.ilocal_id = local_id1;
        ntp_type1.jlocal_id = local_id2;
        maps_type[type1].ntp_attrs.emplace_back(ntp_type1);

        vector1i key = {global_id, type1};
        maps.ntp_global_to_iloc[key] = local_id1;

        if (type1 != type2){
            auto ntp_type2 = ntp;
            ntp_type2.ilocal_id = local_id2;
            ntp_type2.jlocal_id = local_id1;
            maps_type[type2].ntp_attrs.emplace_back(ntp_type2);

            key = {global_id, type2};
            maps.ntp_global_to_iloc[key] = local_id2;
        }
    }
}

void Mapping::set_ntp_local_attrs_compact(){

    auto& maps_type = maps.maps_type;
    const int n_type_pairs = maps.get_n_type_pairs();
    for (int type1 = 0; type1 < n_type; ++type1){
        maps_type[type1].ntp_attrs_tp.resize(n_type_pairs);
        for (const auto& ntp: maps_type[type1].ntp_attrs){
            ntpAttrCompact ntp_compact = {ntp.n_id, ntp.ilocal_id, ntp.jlocal_id};
            maps_type[type1].ntp_attrs_tp[ntp.tp].emplace_back(ntp_compact);
        }
    }
}


void Mapping::set_nlmtp_global_attrs(){

    int global_id(0), global_noconj_id(0), global_conj_id, conj_subtract, n_id;
    const int n_lm = maps.get_n_lm();
    for (int n = 0; n < n_fn; ++n){
        const auto& tp_n = n_to_tp[n];
        for (int lm = 0; lm < n_lm; ++lm){
            const auto& lm_attr = maps.lm_attrs[lm];
            conj_subtract = 2 * lm_attr.m * tp_n.size();
            for (const auto& tp: tp_n){
                global_conj_id = global_id - conj_subtract;
                n_id = tpn_to_n_id[tp][n];
                nlmtpAttr nlmtps = {
                    n, n_id, lm_attr, tp,
                    global_id, global_noconj_id, global_conj_id
                };
                maps.nlmtp_attrs.emplace_back(nlmtps);
                maps.nlmtp_to_global[vector1i({n, lm, tp})] = global_id;
                ++global_id;

                if (lm_attr.conj == false) {
                    maps.nlmtp_attrs_noconj.emplace_back(nlmtps);
                    ++global_noconj_id;
                }
            }
        }
    }

    maps.global_to_global_conj.resize(maps.nlmtp_attrs.size());
    for (const auto& nlmtp: maps.nlmtp_attrs){
        maps.global_to_global_conj[nlmtp.global_id] = nlmtp.global_conj_id;
    }
}

void Mapping::set_nlmtp_local_attrs(){

    auto& maps_type = maps.maps_type;
    for (const auto& nlmtp: maps.nlmtp_attrs){
        const int tp = nlmtp.tp;
        const int type1 = maps.tp_to_types[tp][0];
        const int type2 = maps.tp_to_types[tp][1];

        const int global_id = nlmtp.global_id;
        const int local_id1 = maps_type[type1].nlmtp_attrs.size();
        const int local_id2 = maps_type[type2].nlmtp_attrs.size();
        const int local_noconj_id1 = maps_type[type1].nlmtp_attrs_noconj.size();
        const int local_noconj_id2 = maps_type[type2].nlmtp_attrs_noconj.size();

        auto nlmtp_type1 = nlmtp;
        nlmtp_type1.ilocal_id = local_id1;
        nlmtp_type1.ilocal_noconj_id = local_noconj_id1;
        nlmtp_type1.jlocal_noconj_id = local_noconj_id2;
        maps_type[type1].nlmtp_attrs.emplace_back(nlmtp_type1);
        if (!nlmtp.lm.conj)
            maps_type[type1].nlmtp_attrs_noconj.emplace_back(nlmtp_type1);

        vector1i key = {global_id, type1};
        maps.nlmtp_global_to_iloc[key] = local_id1;

        if (type1 != type2){
            auto nlmtp_type2 = nlmtp;
            nlmtp_type2.ilocal_id = local_id2;
            nlmtp_type2.ilocal_noconj_id = local_noconj_id2;
            nlmtp_type2.jlocal_noconj_id = local_noconj_id1;
            maps_type[type2].nlmtp_attrs.emplace_back(nlmtp_type2);
            if (!nlmtp.lm.conj)
                maps_type[type2].nlmtp_attrs_noconj.emplace_back(nlmtp_type2);

            key = {nlmtp.global_id, type2};
            maps.nlmtp_global_to_iloc[key] = local_id2;
        }
    }
}

void Mapping::set_nlmtp_local_attrs_compact(){

    auto& maps_type = maps.maps_type;
    const int n_type_pairs = maps.get_n_type_pairs();
    for (int type1 = 0; type1 < n_type; ++type1){
        maps_type[type1].nlmtp_attrs_noconj_tp.resize(n_type_pairs);
        const auto& nlmtp_attrs1 = maps_type[type1].nlmtp_attrs_noconj;
        for (const auto& nlmtp: nlmtp_attrs1){
            const int tp = nlmtp.tp;
            nlmtpAttrCompact nlmtp_compact = {
                nlmtp.n_id,
                nlmtp.lm.ylm_key,
                nlmtp.lm.m,
                nlmtp.lm.sign_j,
                nlmtp.ilocal_noconj_id,
                nlmtp.jlocal_noconj_id
            };
            maps_type[type1].nlmtp_attrs_noconj_tp[tp].emplace_back(nlmtp_compact);
        }
    }
}

void Mapping::set_nlmtp_local_conj_ids(){

    for (int t = 0; t < n_type; ++t){
        auto& maps_type = maps.maps_type[t];
        for (auto& nlmtp: maps_type.nlmtp_attrs){
            vector1i key = {nlmtp.global_conj_id, t};
            nlmtp.ilocal_conj_id = maps.nlmtp_global_to_iloc[key];
        }
        for (auto& nlmtp: maps_type.nlmtp_attrs_noconj){
            vector1i key = {nlmtp.global_conj_id, t};
            nlmtp.ilocal_conj_id = maps.nlmtp_global_to_iloc[key];
        }
    }
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
            maps.lm_attrs.emplace_back(lm_attr);
        }
    }
}


Maps& Mapping::get_maps() { return maps; }
