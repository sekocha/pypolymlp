/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_potential_pair.h"

PotentialPair::PotentialPair(){}

PotentialPair::PotentialPair(const feature_params& fp, const vector1d& pot){

    f_obj = FeaturesPair(fp);
    n_type = fp.n_type;

    set_force_terms(pot);
}

PotentialPair::~PotentialPair(){}

void PotentialPair::release_memory(){
    f_obj.release_memory();
}

int PotentialPair::resize_arrays(){

    auto& maps = f_obj.get_maps();

    force_terms1.resize(n_type);
    force_terms2.resize(n_type);
    force_terms3.resize(n_type);
    for (int t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        int n_head_ids = maps_type.ntp_attrs.size();
        force_terms1[t1].resize(n_head_ids);
        force_terms2[t1].resize(n_head_ids);
        force_terms3[t1].resize(n_head_ids);
    }
    return 0;
}

int PotentialPair::set_force_terms(const vector1d& pot){

    resize_arrays();
    auto& maps = f_obj.get_maps();
    for (int t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        const auto& polynomial = maps_type.polynomial;
        for (const auto& term: polynomial){
            const double coeff = pot[term.global_id];
            const int n_prods = term.local_ids.size();
            int ntp_id;
            if (n_prods == 1){
                ntp_id = term.local_ids[0];
                Term1 term_add = {coeff};
                force_terms1[t1][ntp_id].emplace_back(term_add);
            }
            else if (n_prods == 2){
                ntp_id = term.local_ids[0];
                Term2 term_add = {coeff, term.local_ids[1]};
                force_terms2[t1][ntp_id].emplace_back(term_add);

                ntp_id = term.local_ids[1];
                term_add = {coeff, term.local_ids[0]};
                force_terms2[t1][ntp_id].emplace_back(term_add);
            }
            else if (n_prods == 3){
                ntp_id = term.local_ids[0];
                Term3 term_add = {coeff, term.local_ids[1], term.local_ids[2]};
                force_terms3[t1][ntp_id].emplace_back(term_add);

                ntp_id = term.local_ids[1];
                term_add = {coeff, term.local_ids[0], term.local_ids[2]};
                force_terms3[t1][ntp_id].emplace_back(term_add);

                ntp_id = term.local_ids[2];
                term_add = {coeff, term.local_ids[0], term.local_ids[1]};
                force_terms3[t1][ntp_id].emplace_back(term_add);
            }
        }
    }
    return 0;
}

void PotentialPair::compute_sum(
    const vector1d& antp,
    const int type1,
    vector1d& prod_sum_e,
    vector1d& prod_sum_f
){
    const auto& features = antp;
    //vector1d features;
    //f_obj.compute_features(antp, type1, features);

    int ntp_size = force_terms1[type1].size();
    prod_sum_e = vector1d(ntp_size, 0.0);
    prod_sum_f = vector1d(ntp_size, 0.0);

    const auto& fterms1 = force_terms1[type1];
    const auto& fterms2 = force_terms2[type1];
    const auto& fterms3 = force_terms3[type1];
    for (size_t i = 0; i < ntp_size; ++i) {
        double sum_f1 = 0.0;
        double sum_f2 = 0.0;
        double sum_f3 = 0.0;
        for (const auto& term : fterms1[i]) {
            sum_f1 += term.coeff;
        }
        for (const auto& term : fterms2[i]) {
            sum_f2 += term.coeff * features[term.local_id1];
        }
        for (const auto& term : fterms3[i]) {
            sum_f3 += term.coeff
                  * features[term.local_id1]
                  * features[term.local_id2];
        }
        prod_sum_e[i] = 0.5 * (sum_f1 + sum_f2 / 2 + sum_f3 / 3);
        prod_sum_f[i] = 0.5 * (sum_f1 + sum_f2 + sum_f3);
    }
}

int PotentialPair::convert_unit(const double energy_conv){

    for (auto& terms_t: force_terms1){
        for (auto& terms_ntp: terms_t){
            for (auto& term: terms_ntp){
                term.coeff *= energy_conv;
            }
        }
    }
    for (auto& terms_t: force_terms2){
        for (auto& terms_ntp: terms_t){
            for (auto& term: terms_ntp){
                term.coeff *= energy_conv;
            }
        }
    }
    for (auto& terms_t: force_terms3){
        for (auto& terms_ntp: terms_t){
            for (auto& term: terms_ntp){
                term.coeff *= energy_conv;
            }
        }
    }
    return 0;
}

Maps& PotentialPair::get_maps() { return f_obj.get_maps(); }
