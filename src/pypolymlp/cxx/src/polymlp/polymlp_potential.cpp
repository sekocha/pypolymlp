/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        PotentialModel: [PotentialTerm1, PotentialTerm2, ...]

        PotentialTerm: anlmtc[head_key] * prod_anlmtc[prod_key]
                                        * feature[feature_key1]
                                        * feature[feature_key2]
                                        * ...

*****************************************************************************/

#include "polymlp_potential.h"


Potential::Potential(){}

Potential::Potential(const feature_params& fp, const vector1d& pot){

    f_obj = Features(fp, true);
    n_type = fp.n_type;

    auto& maps = f_obj.get_maps();
    if (maps.ntp_attrs.size() > 0) elim_conj = false;
    else elim_conj = true;

    set_terms_using_mapping(pot);
    release_memory();
    sort_potential_model();

}

Potential::~Potential(){}

void Potential::release_memory(){
    f_obj.release_memory();
}

int Potential::set_terms_using_mapping(const vector1d& pot){

    potential_model.resize(n_type);
    prod_features.resize(n_type);

    auto& maps = f_obj.get_maps();
    for (int t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        const auto& features = maps_type.features;
        auto& prod_map_deriv = f_obj.get_prod_map_deriv(t1);
        auto& prod_features_map = f_obj.get_prod_features_map(t1);
        auto& potential_model1 = potential_model[t1];

        std::unordered_map<vector1i, vector1d, HashVI> nonequiv_map;
        for (const auto& term: maps_type.polynomial){
            int n_prods(0);
            for (const int id: term.local_ids){
                n_prods += maps_type.get_feature_size(id);
            }
            for (size_t i = 0; i < term.local_ids.size(); ++i){
                const int head_term_id = term.local_ids[i];
                vector1i term_keys = erase_a_key(term.local_ids, i);
                const int prod_features_id = prod_features_map[term_keys];

                for (const auto& sterm: features[head_term_id]){
                    const double coeff_f = pot[term.global_id] * sterm.coeff;
                    const double coeff_e = coeff_f / double(n_prods);

                    for (int j = 0; j < sterm.nlmtp_ids.size(); ++j){
                        const int head_id = sterm.nlmtp_ids[j];
                        vector1i keys = erase_a_key(sterm.nlmtp_ids, j);
                        const int prod_id = prod_map_deriv[keys];

                        vector1i keys_all = {head_id, prod_id, prod_features_id};
                        if (elim_conj == false or maps_type.is_conj(head_id) == false){
                            if (nonequiv_map.count(keys_all) == 0){
                                nonequiv_map[keys_all] = vector1d{coeff_e,coeff_f};
                            }
                            else {
                                nonequiv_map[keys_all][0] += coeff_e;
                                nonequiv_map[keys_all][1] += coeff_f;
                            }
                        }
                    }
                }
            }
        }

        int n_head_ids;
        if (maps_type.ntp_attrs.size() > 0) n_head_ids = maps_type.ntp_attrs.size();
        else n_head_ids = maps_type.nlmtp_attrs_noconj.size();

        potential_model1.resize(n_head_ids);
        for (const auto& term: nonequiv_map){
            const double coeff_e = term.second[0];
            const double coeff_f = term.second[1];
            const int head_id = term.first[0];
            const int prod_id = term.first[1];
            const int prod_features_id = term.first[2];
            PotentialTerm pterm = {coeff_e, coeff_f, prod_id, prod_features_id};

            if (elim_conj == false){
                potential_model1[head_id].emplace_back(pterm);
            }
            else {
                if (maps_type.is_conj(head_id) == false){
                    const int noconj_id = maps_type.get_noconj_id(head_id);
                    potential_model1[noconj_id].emplace_back(pterm);
                }
            }
        }
    }
    return 0;
}


void Potential::sort_potential_model(){
    // sorted by prod_key and then by prod_features_key
    for (int t1 = 0; t1 < n_type; ++t1){
        for (auto& pmodel: potential_model[t1]){
            std::sort(pmodel.begin(), pmodel.end(),
                    [](const PotentialTerm& lhs, const PotentialTerm& rhs){
                    if (lhs.prod_id != rhs.prod_id){
                        return lhs.prod_id < rhs.prod_id;
                    }
                    else {
                        return lhs.prod_features_id < rhs.prod_features_id;
                    }
                    });
        }
    }
}


void Potential::compute_features(
    const vector1d& antp,
    const int type1,
    vector1d& values
){
    f_obj.compute_features(antp, type1, values);
}


void Potential::compute_features(
    const vector1dc& anlmtp,
    const int type1,
    vector1d& values
){
    f_obj.compute_features(anlmtp, type1, values);
}


void Potential::compute_prod_antp_deriv(
    const vector1d& antp,
    const int type1,
    vector1d& prod_antp_deriv
){
    f_obj.compute_prod_antp_deriv(antp, type1, prod_antp_deriv);
}


void Potential::compute_prod_anlmtp_deriv(
    const vector1dc& anlmtp,
    const int type1,
    vector1dc& prod_anlmtp_deriv
){
    f_obj.compute_prod_anlmtp_deriv(anlmtp, type1, prod_anlmtp_deriv);
}


void Potential::compute_prod_features(
    const vector1d& features,
    const int type1,
    vector1d& values
){
    f_obj.compute_prod_features(features, type1, values);
}


void Potential::compute_sum_of_prod_antp(
    const vector1d& antp,
    const int type1,
    vector1d& prod_sum_e,
    vector1d& prod_sum_f
){
    vector1d features, prod_features_vals, prod_antp_deriv;

    compute_features(antp, type1, features);
    compute_prod_features(features, type1, prod_features_vals);
    compute_prod_antp_deriv(antp, type1, prod_antp_deriv);

    const auto& potential_model1 = potential_model[type1];
    prod_sum_e = vector1d(potential_model1.size());
    prod_sum_f = vector1d(potential_model1.size());

    int i = 0;
    for (const auto& pterms1: potential_model1){
        double sum_e(0.0), sum_f(0.0), prod;
        for (const auto& pterm: pterms1){
            double fval = prod_features_vals[pterm.prod_features_id];
            if (fabs(fval) > 1e-20){
                prod = fval * prod_antp_deriv[pterm.prod_id];
                sum_e += pterm.coeff_e * prod;
                sum_f += pterm.coeff_f * prod;
            }
        }
        prod_sum_e[i] = 0.5 * sum_e;
        prod_sum_f[i] = 0.5 * sum_f;
        ++i;
    }
}


void Potential::compute_sum_of_prod_anlmtp(
    const vector1dc& anlmtp,
    const int type1,
    vector1dc& prod_sum_e,
    vector1dc& prod_sum_f
){
    vector1d features, prod_features_vals;
    vector1dc prod_anlmtp_deriv;
    compute_features(anlmtp, type1, features);
    compute_prod_features(features, type1, prod_features_vals);
    compute_prod_anlmtp_deriv(anlmtp, type1, prod_anlmtp_deriv);

    const auto& potential_model1 = potential_model[type1];
    prod_sum_e = vector1dc(potential_model1.size());
    prod_sum_f = vector1dc(potential_model1.size());

    int i = 0;
    for (const auto& pterms1: potential_model1){
        dc sum_e(0.0), sum_f(0.0);
        for (const auto& pterm: pterms1){
            double fval = prod_features_vals[pterm.prod_features_id];
            if (fabs(fval) > 1e-20){
                sum_e += pterm.coeff_e * fval * prod_anlmtp_deriv[pterm.prod_id];
                sum_f += pterm.coeff_f * fval * prod_anlmtp_deriv[pterm.prod_id];
            }
        }
        prod_sum_e[i] = sum_e;
        prod_sum_f[i] = sum_f;
        ++i;
    }
}

Maps& Potential::get_maps() { return f_obj.get_maps(); }
