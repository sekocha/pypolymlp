/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_api.h"


PolymlpAPI::PolymlpAPI(){}
PolymlpAPI::~PolymlpAPI(){}


int PolymlpAPI::parse_polymlp_file(
    const char *file,
    std::vector<std::string>& ele,
    vector1d& mass
){
    const bool legacy = check_polymlp_legacy(file);
    if (legacy == true) parse_polymlp_legacy(file, fp, pot, ele, mass);
    else parse_polymlp(file, fp, pot, ele, mass);
    return 0;
}


int PolymlpAPI::compute_anlmtp_conjugate(
    const vector1d& anlmtp_r,
    const vector1d& anlmtp_i,
    const int type1,
    vector1dc& anlmtp
){

    const auto& maps = pmodel.get_maps();
    const auto& maps_type = maps.maps_type[type1];
    const auto& nlmtp_attrs = maps_type.nlmtp_attrs;
    const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

    anlmtp = vector1dc(nlmtp_attrs.size(), 0.0);
    int idx(0);
    for (const auto& nlmtp: nlmtp_attrs_noconj){
        const auto& cc_coeff = nlmtp.lm.cc_coeff;
        anlmtp[nlmtp.ilocal_id] = {anlmtp_r[idx], anlmtp_i[idx]};
        anlmtp[nlmtp.ilocal_conj_id] = {
            cc_coeff * anlmtp_r[idx], - cc_coeff * anlmtp_i[idx]
        };
        ++idx;
    }
    return 0;
}


int PolymlpAPI::compute_sum_of_prod_antp(
    const vector1d& antp,
    const int type1,
    vector1d& prod_sum_e,
    vector1d& prod_sum_f
){
    pmodel.compute_sum_of_prod_antp(antp, type1, prod_sum_e, prod_sum_f);
    return 0;
}


int PolymlpAPI::compute_sum_of_prod_anlmtp(
    const vector1dc& anlmtp,
    const int type1,
    vector1dc& prod_sum_e,
    vector1dc& prod_sum_f
){
    pmodel.compute_sum_of_prod_anlmtp(anlmtp, type1, prod_sum_e, prod_sum_f);
    return 0;
}


int PolymlpAPI::set_features(const feature_params& fp){
    features = Features(fp, false);
    return 0;
}


int PolymlpAPI::set_potential_model(){
    pmodel = Potential(fp, pot);
    return 0;
}


const feature_params& PolymlpAPI::get_fp() const { return fp; }
Maps& PolymlpAPI::get_maps() { return pmodel.get_maps(); }
