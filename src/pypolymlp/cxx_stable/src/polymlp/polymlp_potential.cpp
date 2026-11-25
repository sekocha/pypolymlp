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
    flatten_potential_model();

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
                        if (elim_conj and maps_type.is_conj(head_id))
                            continue;

                        vector1i keys = erase_a_key(sterm.nlmtp_ids, j);
                        const int prod_id = prod_map_deriv[keys];

                        vector1i keys_all = {head_id, prod_id, prod_features_id};
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

void Potential::flatten_potential_model(){

    potential_model_flat.resize(n_type);
    offset.resize(n_type);
    for (int t1 = 0; t1 < n_type; ++t1){
        offset[t1].emplace_back(0);
        int cnt(0);
        for (auto& pmodel: potential_model[t1]){
            for (const auto& pterm: pmodel){
                potential_model_flat[t1].emplace_back(pterm);
            }
            cnt += pmodel.size();
            offset[t1].emplace_back(cnt);
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
        double sum_e(0.0), sum_f(0.0);
        for (const auto& pterm: pterms1){
            double fval = prod_features_vals[pterm.prod_features_id];
            double prod = fval * prod_antp_deriv[pterm.prod_id];
            sum_e += pterm.coeff_e * prod;
            sum_f += pterm.coeff_f * prod;
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

    const auto& potential_model1 = potential_model_flat[type1];
    const auto& offset1 = offset[type1];
    const int n_nlmtc_noconj = static_cast<int>(offset1.size()) - 1;

    prod_sum_e = vector1dc(n_nlmtc_noconj);
    prod_sum_f = vector1dc(n_nlmtc_noconj);

    if (n_nlmtc_noconj <= 0) return;

    const PotentialTerm* pbase = potential_model1.data();
    const double* fvals
        = prod_features_vals.empty() ? nullptr : prod_features_vals.data();
    const dc* derivs = prod_anlmtp_deriv.empty() ? nullptr : prod_anlmtp_deriv.data();

    for (int i = 0; i < n_nlmtc_noconj; ++i){
        double sum_e_re = 0.0, sum_e_im = 0.0;
        double sum_f_re = 0.0, sum_f_im = 0.0;

        const int begin = offset1[i];
        const int end   = offset1[i+1];

        #ifdef _OPENMP
        #pragma omp simd reduction(+:sum_e_re,sum_e_im,sum_f_re,sum_f_im)
        #endif
        for (int j = begin; j < end; ++j){
            const PotentialTerm& pt = pbase[j];
            const int pid = pt.prod_id;
            const int pfid = pt.prod_features_id;

            const double fval = fvals[pfid];
            const double s_e = fval * pt.coeff_e; // scalar multiplier for e
            const double s_f = fval * pt.coeff_f; // scalar multiplier for f

            const dc deriv = derivs[pid]; // small copy to allow .real()/.imag()
            const double a = deriv.real();
            const double b = deriv.imag();

            // deriv * s = (a + i b) * s -> real = a*s, imag = b*s
            sum_e_re += a * s_e;
            sum_e_im += b * s_e;
            sum_f_re += a * s_f;
            sum_f_im += b * s_f;
        }

        prod_sum_e[i] = dc(sum_e_re, sum_e_im);
        prod_sum_f[i] = dc(sum_f_re, sum_f_im);
    }
}

/*
    const auto& potential_model1 = potential_model[type1];

    int i = 0;
    for (const auto& pterms1: potential_model1){
        dc sum_e(0.0), sum_f(0.0);
        for (const auto& pterm: pterms1){
            double fval = prod_features_vals[pterm.prod_features_id];
            const dc deriv = prod_anlmtp_deriv[pterm.prod_id];

            double coeff_e = pterm.coeff_e;
            double coeff_f = pterm.coeff_f;
            dc deriv_fval = deriv * fval;
            sum_e += deriv_fval * coeff_e;
            sum_f += deriv_fval * coeff_f;

            //sum_e += deriv * (pterm.coeff_e * fval);
            //sum_f += deriv * (pterm.coeff_f * fval);
            //sum_e += deriv * (coeff_e * fval);
            //sum_f += deriv * (coeff_f * fval);
        }
        prod_sum_e[i] = sum_e;
        prod_sum_f[i] = sum_f;
        ++i;
    }
*/

int Potential::convert_unit(const double energy_conv){
    for (auto& pmodel1: potential_model){
        for (auto& pterms1: pmodel1){
            for (auto& pterm: pterms1){
                pterm.coeff_e *= energy_conv;
                pterm.coeff_f *= energy_conv;
            }
        }
    }
    return 0;
}

Maps& Potential::get_maps() { return f_obj.get_maps(); }
