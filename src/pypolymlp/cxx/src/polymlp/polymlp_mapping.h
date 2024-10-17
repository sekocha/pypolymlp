/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MAPPING
#define __POLYMLP_MAPPING

#include "polymlp_mlpcpp.h"

struct ntpAttr {
    int n;
    int n_id;
    int tp;
    int ntp_key;
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
    int nlmtp_key;
    int conj_key;
    int nlmtp_noconj_key;
};

typedef std::unordered_map<vector1i,int,HashVI> MapFromVec;

class Mapping {

    int n_type, n_type_pairs, n_fn, maxl;
    int n_lm, n_lm_all, n_ntp_all, n_nlmtp_all;
    vector2i type_pairs, map_tp_to_nlist, map_n_to_tplist, n_id_list;
    vector3d map_tp_to_params;

    std::vector<ntpAttr> ntp_attrs;
    std::vector<std::vector<ntpAttr> > ntp_attrs_type;
    std::vector<nlmtpAttr> nlmtp_attrs_no_conjugate, nlmtp_attrs;
    std::vector<lmAttr> lm_attrs;
    MapFromVec map_ntp_to_key, map_nlmtp_to_key;

    void set_type_pairs(const feature_params& fp);
    void set_type_pairs_charge(const feature_params& fp);
    void set_map_n_to_tplist();

    void set_ntp_attrs();
    void set_nlmtp_attrs();
    void set_lm_attrs();

    public:

    Mapping();
    Mapping(const struct feature_params& fp);
    ~Mapping();

    const int get_n_type_pairs() const;
    const int get_n_ntp_all() const;
    const int get_n_nlmtp_all() const;

    // (type1, type2) --> type_pairs
    const vector2i& get_type_pairs() const;
    // type_pair --> conditional n values
    const vector2i& get_type_pair_to_nlist() const;
    // n --> type_pair
    const vector2i& get_n_to_type_pairs() const;
    // (type_pair, n) -> n_id
    const vector2i& get_n_ids() const;
    // type_pair -> params
    const vector3d& get_type_pair_to_params() const;

    const std::vector<ntpAttr>& get_ntp_attrs() const;
    const std::vector<ntpAttr>& get_ntp_attrs(const int type1) const;
    const std::vector<nlmtpAttr>& get_nlmtp_attrs() const;
    const std::vector<nlmtpAttr>& get_nlmtp_attrs_no_conjugate() const;
    const std::vector<lmAttr>& get_lm_attrs() const;

    const MapFromVec& get_ntp_to_key() const;
    const MapFromVec& get_nlmtp_to_key() const;
};

#endif
