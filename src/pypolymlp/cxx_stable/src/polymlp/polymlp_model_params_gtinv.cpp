/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params_gtinv.h"


bool check_type_in_type_pairs(
    const vector1i& tp_comb, const vector2i& type_pairs, const int& type1
){

    const auto& tp_type1 = type_pairs[type1];
    for (const auto& tp: tp_comb){
        if (std::find(tp_type1.begin(), tp_type1.end(), tp) == tp_type1.end()){
            return false;
        }
    }
    return true;
}


vector1i vector_intersection(vector1i v1, vector1i v2){
    vector1i v3;
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    std::set_intersection(
        v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v3)
    );
    return v3;
}


int _find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb){

    const auto iter = std::find(tp_comb_ref.begin(), tp_comb_ref.end(), tp_comb);
    const int index = std::distance(tp_comb_ref.begin(), iter);
    return index;
}


void _enumerate_tp_combs(Maps& maps, const int gtinv_order, vector3i& tp_combs){

    tp_combs.resize(gtinv_order + 1);

    const auto& type_pairs = maps.type_pairs;
    const int n_type = maps.get_n_type();
    const int n_type_pairs = maps.get_n_type_pairs();

    vector1i pinput(n_type_pairs);
    for (int i = 0; i < n_type_pairs; ++i) pinput[i] = i;

    for (int order = 1; order <= gtinv_order; ++order){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, order);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_in_type_pairs(p1, type_pairs, type1) == true){
                    tp_combs[order].emplace_back(p1);
                    break;
                }
            }
        }
    }
}


void _enumerate_nonzero_n(
    Maps& maps, const vector3i& tp_combs, vector3i& nonzero_n_list
){

    const int gtinv_order = tp_combs.size() - 1;
    nonzero_n_list.resize(gtinv_order + 1);
    const auto& tp_to_n = maps.tp_to_n;
    for (int order = 1; order <= gtinv_order; ++order){
        for (const auto& tp_comb: tp_combs[order]){
            vector1i intersection;
            int iter(0);
            for (const auto& tp: tp_comb){
                if (iter == 0) intersection = tp_to_n[tp];
                else intersection = vector_intersection(intersection, tp_to_n[tp]);
                ++iter;
            }
            nonzero_n_list[order].emplace_back(intersection);
        }
    }
}


void uniq_gtinv_type(
    const feature_params& fp,
    Maps& maps,
    vector3i& tp_combs,
    std::vector<LinearTerm>& linear_terms
){

    const int n_fn = fp.params.size();
    const int n_type = fp.n_type;
    const auto& type_pairs = maps.type_pairs;

    const int gtinv_order = (*(fp.l_comb.end() - 1)).size();
    _enumerate_tp_combs(maps, gtinv_order, tp_combs);

    vector3i nonzero_n_list;
    _enumerate_nonzero_n(maps, tp_combs, nonzero_n_list);

    std::vector<std::vector<LinearTerm> > _linear_terms(n_fn);
    for (size_t lm_comb_id = 0; lm_comb_id < fp.l_comb.size(); ++lm_comb_id){
        const vector1i& l_comb = fp.l_comb[lm_comb_id];
        const int order = l_comb.size();
        const auto& tp_combs_ref = tp_combs[order];
        const auto& n_list_ref = nonzero_n_list[order];

        std::set<std::multiset<std::pair<int, int> > > uniq_lmt;
        for (const auto& tp_comb: tp_combs_ref){
            std::multiset<std::pair<int, int> > tmp;
            for (size_t j = 0; j < tp_comb.size(); ++j){
                tmp.insert(std::make_pair(l_comb[j], tp_comb[j]));
            }
            uniq_lmt.insert(tmp);
        }

        for (const auto& lt: uniq_lmt){
            vector1i tp_comb;
            for (const auto& lt1: lt) tp_comb.emplace_back(lt1.second);

            const int tp_comb_id = _find_tp_comb_id(tp_combs_ref, tp_comb);
            const auto& n_list = n_list_ref[tp_comb_id];

            vector1i t1a;
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_in_type_pairs(tp_comb, type_pairs, type1) == true){
                    t1a.emplace_back(type1);
                }
            }

            for (const auto n: n_list){
                const LinearTerm linear = {n, int(lm_comb_id), tp_comb_id, order, t1a};
                _linear_terms[n].emplace_back(linear);
            }
        }
    }

    for (const auto& linear_terms_n: _linear_terms){
        for (const auto& linear: linear_terms_n){
            linear_terms.emplace_back(linear);
        }
    }
}
