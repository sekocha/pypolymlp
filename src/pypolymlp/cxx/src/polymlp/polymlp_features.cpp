/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        SingleTerm: [coeff,[(n1,l1,m1,tc1), (n2,l2,m2,tc2), ...]]
            (n1,l1,m1,tc1) is represented with nlmtc_key.

        SingleFeature: [SingleTerms1, SingleTerms2, ...]

        MultipleFeatures: [SingleFeature1, SingleFeature2, ...]

*****************************************************************************/


#include "polymlp_features.h"

const int threshold_prod = 1000;


Features::Features(){}

Features::Features(const feature_params& fp, const bool set_deriv_i = false){

    n_type = fp.n_type;
    mapping = Mapping(fp);
    auto& maps = mapping.get_maps();
    modelp = ModelParams(fp, maps);

    if (fp.feature_type == "pair") {
        set_deriv = false
        eliminate_conj = false;

        set_linear_features_pair(maps);
        set_mappings_standard();
    }
    else if (fp.feature_type == "gtinv") {
        set_deriv = set_deriv_i;
        eliminate_conj = true;
        if (set_deriv == true) eliminate_conj = false;

        set_linear_features_gtinv(fp, modelp, maps);
        set_mappings_efficient(fp);
    }
    set_deriv_mappings();
    poly = FeaturesPoly(modelp, maps);
}


Features::~Features(){}


int Features::set_mappings_standard(){

    prod.resize(n_type);
    mapped_features.resize(n_type);
    auto& maps = mapping.get_maps();

    for (size_t t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        std::set<vector1i> nonequiv;
        MapFromVec prod_map_from_keys;
        get_nonequiv_ids(maps_type.features, nonequiv);
        convert_set_to_mappings(nonequiv, prod_map_from_keys, prod[t1]);
        convert_to_mapped_features_algo2(
            maps_type.features, prod_map_from_keys, mapped_features[t1]
        );
    }
    return 0;
}


int Features::set_mappings_efficient(const feature_params& fp){

    prod.resize(n_type);
    mapped_features.resize(n_type);
    auto& maps = mapping.get_maps();

    std::vector<MultipleFeatures> features_for_map;
    get_linear_features_gtinv_with_reps(fp, modelp, maps, features_for_map)

    for (size_t t1 = 0; t1 < n_type; ++t1){
        std::set<vector1i> nonequiv;
        MapFromVec prod_map_from_keys;
        get_nonequiv_ids(features_for_map[t1], nonequiv);
        convert_set_to_mappings(nonequiv, prod_map_from_keys, prod[t1]);
        if (prod[t1].size() > threshold_prod){
            convert_to_mapped_features_algo1(
                features_for_map[t1],
                prod_map_from_keys,
                prod[t1],
                mapped_features[t1]
            );
        }
        else {
            convert_to_mapped_features_algo2(
                features_for_map[t1],
                prod_map_from_keys,
                mapped_features[t1]
            );
        }
    }
    return 0;
}

int Features::set_deriv_mappings(){

    prod_deriv.resize(n_type);
    prod_map_deriv_from_keys.resize(n_type);
    auto& maps = mapping.get_maps();
    for (size_t t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        std::set<vector1i> nonequiv;
        get_nonequiv_deriv_ids(features, maps_type, eliminate_conj, nonequiv);
        convert_set_to_mappings(nonequiv, prod_map_deriv_from_keys[t1], prod_deriv[t1]);
    }
    return 0;
}


void Features::compute_features(
    const vector1d& antp,
    const int type1,
    vector1d& feature_values
){
    // for pair
    auto& maps = mapping.get_maps();
    const auto& features1 = maps.maps_type[type1].features;
    const auto& prod1 = prod[type1];
    const auto& mapped_features1 = mapped_features[type1];

    feature_values = vector1d(features1.size(), 0.0);
    for (size_t i = 0; i < mapped_features1.size(); ++i){
        const auto& sterm = mapped_features1[i][0];
        feature_values[i] = antp[prod1[sterm.id][0]];
    }
}


void Features::compute_features(
    const vector1dc& anlmtp,
    const int type1,
    vector1d& feature_values
){
    // for gtinv
    auto& maps = mapping.get_maps();
    const auto& features1 = maps.maps_type[type1].features;
    const auto& prod1 = prod[type1];
    const auto& mapped_features1 = mapped_features[type1];

    feature_values = vector1d(features1.size(), 0.0);
    if (prod1.size() > threshold_prod){
        for (size_t i = 0; i < mapped_features1.size(); ++i){
            double val = compute_product_real(prod1[i], anlmtp);
            if (fabs(val) > 1e-20){
                for (const auto& mf: mapped_features1[i]){
                    feature_values[mf.id] += mf.coeff * val;
                }
            }
        }
    }
    else {
        vector1d prod_anlmtp;
        compute_products_real(prod1, anlmtp, prod_anlmtp);
        double val;
        for (size_t i = 0; i < mapped_features1.size(); ++i){
            val = 0.0;
            for (const auto& sterm: mapped_features1[i]){
                val += sterm.coeff * prod_anlmtp[sterm.id];
            }
            feature_values[i] = val;
        }
    }
}


void Features::compute_features_deriv(
    const vector2dc& anlmtp_d,
    const int type1,
    vector2d& derivs
){
    // Derivatives of features for gtinv

    // a : local_a_size x n_atom
    // deriv: localsize x n_atom
    auto& maps = mapping.get_maps();
    const auto& features1 = maps.maps_type[type1].features;
    derivs = vector2d(features1.size(), vector1d(anlmtp_d[0].size(), 0.0));

    const auto& prod1 = prod[type1];
    const auto& mapped_features1 = mapped_features[type1];

    for (size_t i = 0; i < mapped_features1.size(); ++i){
        double val = compute_product_real(prod1[i], anlmtp);
        if (fabs(val) > 1e-20){
            for (const auto& mf: mapped_features1[i]){
                feature_values[mf.id] += mf.coeff * val;
            }
        }
    }
}


void Features::compute_prod_antp_deriv(
    const vector1d& antp,
    const int type1,
    vector1d& prod_antp_deriv
){
    compute_products<double>(prod_deriv[type1], antp, prod_antp_deriv);
}


void Features::compute_prod_anlmtp_deriv(
    const vector1dc& anlmtp,
    const int type1,
    vector1dc& prod_anlmtp_deriv
){
    compute_products<dc>(prod_deriv[type1], anlmtp, prod_anlmtp_deriv);
}


void Features::compute_prod_features(
    const vector1d& features,
    const int type1,
    vector1d& values
){
    poly.compute_prod_features(features, type1, values);
}


Maps& Features::get_maps() { return mapping.get_maps(); }


MapFromVec& Features::get_prod_map_deriv(const int type1){
    return prod_map_deriv_from_keys[type1];
}


MapFromVec& Features::get_prod_features_map(const int type1){
    return poly.get_prod_features_map(type1);
}

const int Features::get_n_variables() const {
    return poly.get_n_variables();
}
