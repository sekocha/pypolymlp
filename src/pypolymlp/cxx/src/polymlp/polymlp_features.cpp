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
        set_features_deriv = false;
        eliminate_conj = false;

        set_linear_features_pair(maps);
        set_mappings_standard();
    }
    else if (fp.feature_type == "gtinv") {
        set_features_deriv = set_deriv_i;
        if (set_features_deriv == true) eliminate_conj = false;
        else eliminate_conj = true;

        set_linear_features_gtinv(fp, modelp, maps);
        set_mappings_efficient(fp);
    }
    set_deriv_mappings();
    poly = FeaturesPoly(modelp, maps);

    if (set_features_deriv == true) set_mapped_features_deriv();

    for (int type1 = 0; type1 < n_type; ++type1){
        const auto& features1 = maps.maps_type[type1].features;
        feature_sizes.emplace_back(features1.size());
    }
}


Features::~Features(){}

void Features::release_memory(){
    auto& maps = mapping.get_maps();
    maps.clear();
    prod_map_deriv_from_keys = {};
}

int Features::set_mappings_standard(){

    prod.resize(n_type);
    mapped_features.resize(n_type);
    auto& maps = mapping.get_maps();

    for (size_t t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        std::set<vector1i> nonequiv;
        get_nonequiv_ids(maps_type.features, nonequiv);

        MapFromVec prod_map_from_keys;
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

    std::vector<MultipleFeatures> features_for_map(n_type);
    get_linear_features_gtinv_with_reps(fp, modelp, maps, features_for_map);

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
        get_nonequiv_deriv_ids(maps_type.features, maps_type, eliminate_conj, nonequiv);
        convert_set_to_mappings(nonequiv, prod_map_deriv_from_keys[t1], prod_deriv[t1]);
    }
    return 0;
}


int Features::set_mapped_features_deriv(){

    mapped_features_deriv.resize(n_type);
    auto& maps = mapping.get_maps();
    for (size_t t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        convert_to_mapped_features_deriv_algo2(
            maps_type.features,
            prod_map_deriv_from_keys[t1],
            mapped_features_deriv[t1]
        );
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
    const auto& prod1 = prod[type1];
    const auto& mapped_features1 = mapped_features[type1];

    feature_values = vector1d(feature_sizes[type1], 0.0);
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
    const auto& prod1 = prod[type1];
    const auto& mapped_features1 = mapped_features[type1];

    feature_values = vector1d(feature_sizes[type1], 0.0);
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
    //Calculate derivatives of gtinv features.
    vector1dc prod_anlmtp_d;
    compute_prod_anlmtp_deriv(anlmtp, type1, prod_anlmtp_d);
    compute_features_deriv_single_component(anlmtp_dfx, type1, prod_anlmtp_d, dn_dfx);
    compute_features_deriv_single_component(anlmtp_dfy, type1, prod_anlmtp_d, dn_dfy);
    compute_features_deriv_single_component(anlmtp_dfz, type1, prod_anlmtp_d, dn_dfz);
    compute_features_deriv_single_component(anlmtp_ds, type1, prod_anlmtp_d, dn_ds);
/*
    const auto& mapped_features_deriv1 = mapped_features_deriv[type1];
    const int n_atom = anlmtp_dfx[0].size();
    dc val;
    for (const auto& mf: mapped_features_deriv1){
        vector1d dx(n_atom, 0.0), dy(n_atom, 0.0), dz(n_atom, 0.0), ds(6, 0.0);
        for (const auto& sterm: mf){
            val = sterm.coeff * prod_anlmtp_d[sterm.prod_id];
            for (int j = 0; j < n_atom; ++j)
                dx[j] += prod_real(val, anlmtp_dfx[sterm.head_id][j]);
            for (int j = 0; j < n_atom; ++j)
                dy[j] += prod_real(val, anlmtp_dfy[sterm.head_id][j]);
            for (int j = 0; j < n_atom; ++j)
                dz[j] += prod_real(val, anlmtp_dfz[sterm.head_id][j]);
            for (int j = 0; j < 6; ++j)
                ds[j] += prod_real(val, anlmtp_ds[sterm.head_id][j]);
        }
        dn_dfx.emplace_back(std::move(dx));
        dn_dfy.emplace_back(std::move(dy));
        dn_dfz.emplace_back(std::move(dz));
        dn_ds.emplace_back(std::move(ds));
    }
    */
}

void Features::compute_features_deriv_single_component(
    const vector2dc& anlmtp_d,
    const int type1,
    const vector1dc& prod_anlmtp_d,
    vector2d& dn_d
){

    const auto& mapped_features_deriv1 = mapped_features_deriv[type1];
    const size_t n_col = anlmtp_d[0].size();
    dc val;
    for (const auto& mf: mapped_features_deriv1){
        vector1d dvals(n_col, 0.0);
        for (const auto& sterm: mf){
            val = sterm.coeff * prod_anlmtp_d[sterm.prod_id];
            for (size_t j = 0; j < n_col; ++j)
                dvals[j] += prod_real(val, anlmtp_d[sterm.head_id][j]);
        }
        dn_d.emplace_back(std::move(dvals));
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
