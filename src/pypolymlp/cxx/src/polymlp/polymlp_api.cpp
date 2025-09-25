/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_api.h"


PolymlpAPI::PolymlpAPI(){
    use_features = false;
    use_potential = false;
    use_model_params = false;
}

PolymlpAPI::~PolymlpAPI(){}


int PolymlpAPI::parse_polymlp_file(
    const char *file,
    std::vector<std::string>& ele,
    vector1d& mass
){
    use_potential = true;
    const bool legacy = check_polymlp_legacy(file);
    if (legacy == true) parse_polymlp_legacy(file, fp, pot, ele, mass);
    else parse_polymlp(file, fp, pot, ele, mass);

    pmodel = Potential(fp, pot);
    return 0;
}


int PolymlpAPI::set_potential_model(const feature_params& fp_i, const vector1d& pot_i){
    use_potential = true;
    fp = fp_i;
    pot = pot_i;
    pmodel = Potential(fp, pot);
    return 0;
}


int PolymlpAPI::set_features(const feature_params& fp_i){
    use_features = true;
    fp = fp_i;
    const bool set_deriv = true;
    features = Features(fp, set_deriv);
    features.release_memory();
    return 0;
}

int PolymlpAPI::set_model_parameters(const feature_params& fp_i){
    use_model_params = true;
    fp = fp_i;
    mapping = Mapping(fp);
    auto& maps = mapping.get_maps();
    modelp = ModelParams(fp, maps);
    return 0;
}


int PolymlpAPI::compute_anlmtp_conjugate(
    const vector1d& anlmtp_r,
    const vector1d& anlmtp_i,
    const int type1,
    vector1dc& anlmtp
){

    const auto& maps = get_maps();
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


int PolymlpAPI::compute_anlmtp_conjugate(
    const vector2d& anlmtp_r,
    const vector2d& anlmtp_i,
    const int type1,
    vector2dc& anlmtp
){

    const auto& maps = get_maps();
    const auto& maps_type = maps.maps_type[type1];
    const auto& nlmtp_attrs = maps_type.nlmtp_attrs;
    const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

    const int n_col = anlmtp_r[0].size();
    anlmtp = vector2dc(nlmtp_attrs.size(), vector1dc(n_col, 0.0));
    int idx(0);
    double rval, ival;
    for (const auto& nlmtp: nlmtp_attrs_noconj){
        const auto& cc_coeff = nlmtp.lm.cc_coeff;
        auto& anlmtp1 = anlmtp[nlmtp.ilocal_id];
        auto& anlmtp2 = anlmtp[nlmtp.ilocal_conj_id];
        for (size_t k = 0; k < n_col; ++k){
            rval = anlmtp_r[idx][k];
            ival = anlmtp_i[idx][k];
            anlmtp1[k] = {rval, ival};
            anlmtp2[k] = {cc_coeff * rval, - cc_coeff * ival};
            //anlmtp[nlmtp.ilocal_id][k] = {rval, ival};
            //anlmtp[nlmtp.ilocal_conj_id][k] = {cc_coeff * rval, - cc_coeff * ival};
        }
        ++idx;
    }
    return 0;
}


int PolymlpAPI::compute_features(
    const vector1d& antp,
    const int type1,
    vector1d& feature_values
){
    features.compute_features(antp, type1, feature_values);
    return 0;
}


int PolymlpAPI::compute_features(
    const vector1dc& anlmtp,
    const int type1,
    vector1d& feature_values
){
    features.compute_features(anlmtp, type1, feature_values);
    return 0;
}

int PolymlpAPI::compute_features_deriv(
    const vector1dc& anlmtp,
    const vector2dc& anlmtp_dfx,
    const vector2dc& anlmtp_dfy,
    const vector2dc& anlmtp_dfz,
    const vector2dc& anlmtp_ds,
    const int type1,
    vector2d& dn_dfx,
    vector2d& dn_dfy,
    vector2d& dn_dfz,
    vector2d& dn_ds
){
    features.compute_features_deriv(
        anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds, type1,
        dn_dfx, dn_dfy, dn_dfz, dn_ds
    );
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


const feature_params& PolymlpAPI::get_fp() const { return fp; }
const ModelParams& PolymlpAPI::get_model_params() const { return modelp; }

Maps& PolymlpAPI::get_maps() {
    if (use_potential) return pmodel.get_maps();
    else if (use_model_params) return mapping.get_maps();
    return features.get_maps();
}

int PolymlpAPI::get_n_variables() {
    if (use_features) return features.get_n_variables();
    else
        std::cerr << "No method is found for getting n_variables." << std::endl;
        exit(8);
}
