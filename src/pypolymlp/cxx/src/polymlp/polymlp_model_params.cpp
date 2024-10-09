/****************************************************************************
        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params.h"

ModelParams::ModelParams(){}
ModelParams::ModelParams(const feature_params& fp, const Mapping& mapping){

    n_type = fp.n_type;
    type_pairs = mapping.get_type_pairs();
    initial_setting(fp, mapping);
}

ModelParams::~ModelParams(){}


void ModelParams::initial_setting(const feature_params& fp, const Mapping& mapping){

    n_type = fp.n_type;
    n_fn = fp.params.size();

    if (fp.feature_type == "pair"){
        n_linear_features = mapping.get_ntp_attrs().size();
    }
    else if (fp.feature_type == "gtinv"){
        uniq_gtinv_type(fp, mapping);
        n_linear_features = linear_terms.size();
    }

    vector1i polynomial_indices;
    if (fp.model_type == 2){
        for (int n = 0; n < n_linear_features; ++n){
            polynomial_indices.emplace_back(n);
        }
    }
    else if (fp.model_type == 3 and fp.feature_type == "gtinv"){
        for (int n = 0; n < n_linear_features; ++n){
            if (linear_terms[n].order == 1) polynomial_indices.emplace_back(n);
        }
    }
    else if (fp.model_type == 4 and fp.feature_type == "gtinv"){
        for (int n = 0; n < n_linear_features; ++n){
            if (linear_terms[n].order < 3) polynomial_indices.emplace_back(n);
        }
    }

    comb1_indices.resize(n_type);
    comb2_indices.resize(n_type);
    comb3_indices.resize(n_type);

    if (fp.feature_type == "pair") combination1();
    else if (fp.feature_type == "gtinv") combination1_gtinv();

    if (fp.model_type == 1) n_coeff_all = n_linear_features * fp.maxp;
    else if (fp.model_type > 1){
        if (fp.feature_type == "pair"){
            if (fp.maxp > 1) combination2(polynomial_indices);
            if (fp.maxp > 2) combination3(polynomial_indices);
        }
        else if (fp.feature_type == "gtinv"){
            if (fp.maxp > 1) combination2_gtinv(polynomial_indices);
            if (fp.maxp > 2) combination3_gtinv(polynomial_indices);
        }
        n_coeff_all = n_linear_features + comb2.size() + comb3.size();
    }
}


void ModelParams::enumerate_tp_combs(const feature_params& fp, const Mapping& mapping){

    const vector2i& l_comb = fp.l_comb;
    const int gtinv_order = (*(l_comb.end() - 1)).size();

    vector1i pinput(n_type_pairs);
    for (int i = 0; i < n_type_pairs; ++i) pinput[i] = i;

    tp_combs.resize(gtinv_order + 1);
    for (int order = 1; order <= gtinv_order; ++order){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, order);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_pairs(p1, type1) == true){
                    tp_combs[order].emplace_back(p1);
                    break;
                }
            }
        }
    }

    const auto& tp_nlist_map = mapping.get_type_pair_to_nlist();
    params_conditional.resize(gtinv_order + 1);
    for (int order = 1; order <= gtinv_order; ++order){
        for (const auto& tp_comb: tp_combs[order]){
            vector1i intersection;
            int iter = 0;
            for (const auto& tp: tp_comb){
                if (iter == 0) intersection = tp_nlist_map[tp];
                else {
                    intersection = vector_intersection(
                        intersection, tp_nlist_map[tp]
                    );
                }
                ++iter;
            }
            params_conditional[order].emplace_back(intersection);
        }
    }
}

vector1i ModelParams::vector_intersection(vector1i v1, vector1i v2){
    vector1i v3;
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    std::set_intersection(
        v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v3)
    );
    return v3;
}


int ModelParams::find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb){

    const auto iter = std::find(tp_comb_ref.begin(), tp_comb_ref.end(), tp_comb);
    const int index = std::distance(tp_comb_ref.begin(), iter);
    return index;
}

void ModelParams::uniq_gtinv_type(const feature_params& fp, const Mapping& mapping){

    enumerate_tp_combs(fp, mapping);
    std::vector<std::vector<LinearTerm> > _linear_terms(n_fn);

    for (size_t lm_comb_id = 0; lm_comb_id < fp.l_comb.size(); ++lm_comb_id){
        const vector1i& l_comb = fp.l_comb[lm_comb_id];
        const auto& lm_list = fp.lm_array[lm_comb_id];
        const auto& coeff_list = fp.lm_coeffs[lm_comb_id];
        const int order = l_comb.size();

        const auto& tp_combs_ref = tp_combs[order];
        const auto& n_list_ref = params_conditional[order];

        std::set<std::multiset<std::pair<int, int> > > uniq_lmt;
        for (const auto &tp_comb: tp_combs_ref){
            std::multiset<std::pair<int, int> > tmp;
            for (size_t j = 0; j < tp_comb.size(); ++j){
                tmp.insert(std::make_pair(l_comb[j], tp_comb[j]));
            }
            uniq_lmt.insert(tmp);
        }

        for (const auto& lt: uniq_lmt){
            vector1i tp_comb;
            for (const auto& lt1: lt) tp_comb.emplace_back(lt1.second);

            const int tp_comb_id = find_tp_comb_id(tp_combs_ref, tp_comb);
            const auto& n_list = n_list_ref[tp_comb_id];
            vector1i t1a;
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_pairs(tp_comb, type1) == true)
                    t1a.emplace_back(type1);
            }

            for (int n_id = 0; n_id < n_list.size(); ++n_id){
                const LinearTerm linear = {
                    n_list[n_id], n_id, int(lm_comb_id), tp_comb_id, order, t1a
                };
                _linear_terms[n_list[n_id]].emplace_back(linear);
            }
        }
    }

    for (const auto& linear_terms_n: _linear_terms){
        for (const auto& linear: linear_terms_n){
            linear_terms.emplace_back(linear);
        }
    }
}

void ModelParams::combination1_gtinv(){

    for (size_t i_comb = 0; i_comb < linear_terms.size(); ++i_comb){
        for (const auto& type: linear_terms[i_comb].type1){
            comb1_indices[type].emplace_back(int(i_comb));
        }
    }
}

void ModelParams::combination2_gtinv(const vector1i& iarray){

    vector2i type_array;
    vector1i intersection;
    int i_comb(0);
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        const auto& type1_1 = linear_terms[i1].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[i2].type1;
            type_array = {type1_1, type1_2};
            intersection = intersection_types_in_polynomial(type_array);
            if (intersection.size() > 0){
                comb2.push_back(vector1i({iarray[i2], iarray[i1]}));
                for (const auto& type: intersection){
                    comb2_indices[type].emplace_back(i_comb);
                }
                ++i_comb;
            }
        }
    }
}

void ModelParams::combination3_gtinv(const vector1i& iarray){

    vector2i type_array;
    vector1i intersection;
    int i_comb(0);
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        const auto& type1_1 = linear_terms[i1].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[i2].type1;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                const auto& type1_3 = linear_terms[i3].type1;
                type_array = {type1_1, type1_2, type1_3};
                intersection = intersection_types_in_polynomial(type_array);
                if (intersection.size() > 0){
                    comb3.push_back
                        (vector1i({iarray[i3], iarray[i2], iarray[i1]}));
                    for (const auto& type: intersection){
                        comb3_indices[type].emplace_back(i_comb);
                    }
                    ++i_comb;
                }
            }
        }
    }
}

vector1i ModelParams::intersection_types_in_polynomial(
    const vector2i &type1_array
){

    vector2i type1_array_sorted;
    for (const auto& t1: type1_array){
        vector1i t1_copy(t1);
        std::sort(t1_copy.begin(), t1_copy.end());
        type1_array_sorted.emplace_back(t1);
    }

    vector1i intersection(type1_array_sorted[0]);
    for (size_t i = 1; i < type1_array_sorted.size(); ++i){
        vector1i intersection_tmp;
        std::set_intersection(
            intersection.begin(), intersection.end(),
            type1_array_sorted[i].begin(), type1_array_sorted[i].end(),
            back_inserter(intersection_tmp));
        intersection = intersection_tmp;
    }
    return intersection;
}

int ModelParams::seq2typecomb(const int& seq){
    return seq/n_fn;
}
/*
int ModelParams::seq2igtinv(const int& seq){
    return seq % linear_array_g.size();
}
*/

void ModelParams::combination1(){

    int t1;
    for (int i = 0; i < n_linear_features; ++i){
        t1 = seq2typecomb(i);
        for (int type1 = 0; type1 < n_type; ++type1){
            if (check_type_pairs(vector1i({t1}), type1) == true){
                comb1_indices[type1].emplace_back(i);
            }
        }
    }
}


void ModelParams::combination2(const vector1i& iarray){

    int i_comb(0), t1, t2;
    bool match;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        t1 = seq2typecomb(iarray[i1]);
        for (size_t i2 = 0; i2 <= i1; ++i2){
            t2 = seq2typecomb(iarray[i2]);
            match = false;
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_pairs(vector1i({t1,t2}), type1) == true){
                    comb2_indices[type1].emplace_back(i_comb);
                    match = true;
                }
            }
            if (match == true) {
                comb2.push_back(vector1i({iarray[i2],iarray[i1]}));
                ++i_comb;
            }
        }
    }
}

void ModelParams::combination3(const vector1i& iarray){

    int i_comb(0), t1, t2, t3;
    bool match;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        t1 = seq2typecomb(iarray[i1]);
        for (size_t i2 = 0; i2 <= i1; ++i2){
            t2 = seq2typecomb(iarray[i2]);
            for (size_t i3 = 0; i3 <= i2; ++i3){
                t3 = seq2typecomb(iarray[i3]);
                match = false;
                for (int type1 = 0; type1 < n_type; ++type1){
                    if (check_type_pairs
                        (vector1i({t1,t2,t3}), type1) == true){
                        comb3_indices[type1].emplace_back(i_comb);
                        match = true;

                    }
                }
                if (match == true) {
                    comb3.push_back
                        (vector1i({iarray[i3],iarray[i2],iarray[i1]}));
                    ++i_comb;
                }
            }
        }
    }
}

bool ModelParams::check_type_pairs(const vector1i& tp_comb, const int& type1) const{

    const auto& tp_type1 = type_pairs[type1];
    for (const auto& tp: tp_comb){
        if (std::find(tp_type1.begin(), tp_type1.end(), tp) == tp_type1.end()){
            return false;
        }
    }
    return true;
}

//const int& ModelParams::get_n_type() const { return n_type; }
//const int& ModelParams::get_n_type_pairs() const { return n_type_pairs; }
//const int& ModelParams::get_n_fn() const { return n_fn; }

const int& ModelParams::get_n_linear_features() const { return n_linear_features; }
const int& ModelParams::get_n_coeff_all() const { return n_coeff_all; }
const vector2i& ModelParams::get_comb2() const { return comb2; }
const vector2i& ModelParams::get_comb3() const{ return comb3; }

const vector1i& ModelParams::get_comb1_indices(const int type) const {
    return comb1_indices[type];
}
const vector1i& ModelParams::get_comb2_indices(const int type) const {
    return comb2_indices[type];
}
const vector1i& ModelParams::get_comb3_indices(const int type) const {
    return comb3_indices[type];
}

const std::vector<struct LinearTermGtinv>& ModelParams::get_linear_term_gtinv() const{
    return linear_array_g;
}
const std::vector<struct LinearTerm>& ModelParams::get_linear_terms() const{
    return linear_terms;
}
const vector3i& ModelParams::get_tp_combs() const{
    return tp_combs;
}

/*
    int order = 0;
    for (auto& v1: tp_comb_candidates){
        std::cout << "order = " << order << std::endl;
        for (auto& v2: v1){
            for (auto& v3: v2){
                std::cout << v3 << " ";
            }
            std::cout << std::endl;
        }
    }
    */
