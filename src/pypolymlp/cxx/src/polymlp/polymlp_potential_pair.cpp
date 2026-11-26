/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

/*****************************************************************************

        PotentialPairModel: [PotentialPairTerm1, PotentialPairTerm2, ...]

        PotentialPairTerm: anlmtc[head_key] * prod_anlmtc[prod_key]
                                        * feature[feature_key1]
                                        * feature[feature_key2]
                                        * ...

*****************************************************************************/

#include "polymlp_potential_pair.h"

PotentialPair::PotentialPair(){}

PotentialPair::PotentialPair(const feature_params& fp, const vector1d& pot){

    f_obj = FeaturesPair(fp);
    n_type = fp.n_type;

    auto& maps = f_obj.get_maps();
    set_force_terms(pot);
}

PotentialPair::~PotentialPair(){}

//void PotentialPair::release_memory(){
//    f_obj.release_memory();
//}

int PotentialPair::set_force_terms(const vector1d& pot){

    force_terms1.resize(n_type);
    force_terms2.resize(n_type);
    force_terms3.resize(n_type);
    auto& maps = f_obj.get_maps();
    for (int t1 = 0; t1 < n_type; ++t1){
        auto& maps_type = maps.maps_type[t1];
        const auto& polynomial = maps_type.polynomial;
        for (const auto& term: polynomial){
            const double coeff = pot[term.global_id];
            const int n_prods = term.local_ids.size();
            if (n_prods == 1){
                force_terms1[t1].emplace_back(ForceTerm1(coeff));
            }
            else if (n_prods == 2){
                force_terms2[t1].emplace_back(ForceTerm2(coeff, term.local_ids[0]));
                force_terms2[t1].emplace_back(ForceTerm2(coeff, term.local_ids[1]));
            }
            else if (n_prods == 3){
                force_terms3[t1].emplace_back(
                    ForceTerm3(coeff, term.local_ids[0], term.local_ids[1]));
                force_terms3[t1].emplace_back(
                    ForceTerm3(coeff, term.local_ids[0], term.local_ids[2]));
                force_terms3[t1].emplace_back(
                    ForceTerm3(coeff, term.local_ids[1], term.local_ids[2]));
            }
        }
    }
}

void PotentialPair::compute_sum(
    const vector1d& antp,
    const int type1,
    vector1d& prod_sum_e,
    vector1d& prod_sum_f
){
    vector1d features;
    f_obj.compute_features(antp, type1, features);

    //compute_prod_features(features, type1, prod_features_vals);
    //compute_prod_antp_deriv(antp, type1, prod_antp_deriv);
    double sum_f(0.0);
    for (const auto& term: force_terms1[type1]){
        sum_f += term.coeff;
    }
    for (const auto& term: force_terms2[type1]){
        sum_f += term.coeff * features[term.local_id1];
    }
    for (const auto& term: force_terms3[type1]){
        sum_f += term.coeff * features[term.local_id1] * features[term.local_id2];
    }
    //prod_sum_f[i] = 0.5 * sum_f;
    //++i;
}

//int PotentialPair::convert_unit(const double energy_conv){
//    for (auto& pmodel1: potential_model){
//        for (auto& pterms1: pmodel1){
//            for (auto& pterm: pterms1){
//                pterm.coeff_e *= energy_conv;
//                pterm.coeff_f *= energy_conv;
//            }
//        }
//    }
//    return 0;
//}

Maps& PotentialPair::get_maps() { return f_obj.get_maps(); }
