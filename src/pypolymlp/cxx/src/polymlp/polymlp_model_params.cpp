/****************************************************************************
        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params.h"

ModelParams::ModelParams(){}
ModelParams::ModelParams(const feature_params& fp, const Mapping& mapping){

    n_type = fp.n_type;
    n_fn = fp.params.size();
    type_pairs = mapping.get_type_pairs();
    n_type_pairs = mapping.get_n_type_pairs();

    initial_setting(fp, mapping);
}

ModelParams::~ModelParams(){}

void ModelParams::initial_setting(const feature_params& fp, const Mapping& mapping){

    if (fp.feature_type == "pair"){
        n_linear_features = mapping.get_ntp_attrs().size();
    }
    else if (fp.feature_type == "gtinv"){
        uniq_gtinv_type(fp, mapping);
        n_linear_features = linear_terms.size();
    }
    polynomial_setting(fp, mapping);
}

void ModelParams::polynomial_setting(const feature_params& fp, const Mapping& mapping){

    vector1i polynomial_indices;
    if (fp.model_type == 2 or fp.model_type == 12){
        for (int n = 0; n < n_linear_features; ++n){
            polynomial_indices.emplace_back(n);
        }
    }
    else if (fp.feature_type == "pair" and fp.model_type > 2){
        std::cerr << "Polymlp: Model type error." << std::endl;
        exit(8);
    }
    else if (fp.feature_type == "gtinv"){
        if (fp.model_type == 3 or fp.model_type == 13){
            for (int n = 0; n < n_linear_features; ++n){
                if (linear_terms[n].order == 1) polynomial_indices.emplace_back(n);
            }
        }
        else if (fp.model_type == 4 or fp.model_type == 14){
            for (int n = 0; n < n_linear_features; ++n){
                if (linear_terms[n].order < 3) polynomial_indices.emplace_back(n);
            }
        }
        else if (fp.model_type > 4){
            std::cerr << "Polymlp: Model type error." << std::endl;
            exit(8);
        }
    }
    if (fp.model_type > 11) find_active_clusters(fp);

    comb1_indices.resize(n_type);
    comb2_indices.resize(n_type);
    comb3_indices.resize(n_type);

    if (fp.feature_type == "pair"){
        combination1(mapping);
        if (fp.model_type > 1 and fp.model_type < 5){
            if (fp.maxp > 1) combination2(polynomial_indices, mapping);
            if (fp.maxp > 2) combination3(polynomial_indices, mapping);
        }
        else if (fp.model_type > 11){
            if (fp.maxp > 1) combination2_cutoff(polynomial_indices, mapping);
            if (fp.maxp > 2) combination3_cutoff(polynomial_indices, mapping);
        }
    }
    else if (fp.feature_type == "gtinv"){
        combination1_gtinv();
        if (fp.model_type > 1 and fp.model_type < 5){
            if (fp.maxp > 1) combination2_gtinv(polynomial_indices);
            if (fp.maxp > 2) combination3_gtinv(polynomial_indices);
        }
        else if (fp.model_type > 11){
            if (fp.maxp > 1) combination2_gtinv_cutoff(polynomial_indices);
            if (fp.maxp > 2) combination3_gtinv_cutoff(polynomial_indices);
        }
    }

    if (fp.model_type == 1) n_coeff_all = n_linear_features * fp.maxp;
    else n_coeff_all = n_linear_features + comb2.size() + comb3.size();
}

void ModelParams::enumerate_tp_combs(const int gtinv_order){

    tp_combs.resize(gtinv_order + 1);

    vector1i pinput(n_type_pairs);
    for (int i = 0; i < n_type_pairs; ++i) pinput[i] = i;

    for (int order = 1; order <= gtinv_order; ++order){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, order);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_in_type_pairs(p1, type1) == true){
                    tp_combs[order].emplace_back(p1);
                    break;
                }
            }
        }
    }
}

void ModelParams::enumerate_nonzero_n(const Mapping& mapping){

    const int gtinv_order = tp_combs.size() - 1;
    nonzero_n_list.resize(gtinv_order + 1);
    const auto& tp_to_nlist = mapping.get_type_pair_to_nlist();
    for (int order = 1; order <= gtinv_order; ++order){
        for (const auto& tp_comb: tp_combs[order]){
            vector1i intersection;
            int iter(0);
            for (const auto& tp: tp_comb){
                if (iter == 0) intersection = tp_to_nlist[tp];
                else intersection = vector_intersection(intersection, tp_to_nlist[tp]);
                ++iter;
            }
            nonzero_n_list[order].emplace_back(intersection);
        }
    }
}

bool ModelParams::check_type_in_type_pairs(
    const vector1i& tp_comb, const int& type1
) const {

    const auto& tp_type1 = type_pairs[type1];
    for (const auto& tp: tp_comb){
        if (std::find(tp_type1.begin(), tp_type1.end(), tp) == tp_type1.end()){
            return false;
        }
    }
    return true;
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


void ModelParams::uniq_gtinv_type(const feature_params& fp, const Mapping& mapping){

    const vector2i& l_comb = fp.l_comb;
    const int gtinv_order = (*(l_comb.end() - 1)).size();
    enumerate_tp_combs(gtinv_order);
    enumerate_nonzero_n(mapping);

    std::vector<std::vector<LinearTerm> > _linear_terms(n_fn);
    for (size_t lm_comb_id = 0; lm_comb_id < fp.l_comb.size(); ++lm_comb_id){
        const vector1i& l_comb = fp.l_comb[lm_comb_id];
        const int order = l_comb.size();
        const auto& tp_combs_ref = tp_combs[order];
        const auto& n_list_ref = nonzero_n_list[order];

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
                if (check_type_in_type_pairs(tp_comb, type1) == true){
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

int ModelParams::find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb){

    const auto iter = std::find(tp_comb_ref.begin(), tp_comb_ref.end(), tp_comb);
    const int index = std::distance(tp_comb_ref.begin(), iter);
    return index;
}

/*
void ModelParams::find_active_clusters(const feature_params& fp){

    if (fp.maxp > 1){
        double cutoff = 0.0;
        for (const auto& p: fp.params) cutoff += p[0];
        cutoff *= 2.5 / fp.params.size();

        vector1i narray;
        for (int n1 = 0; n1 < fp.params.size(); ++n1){
            for (int n2 = 0; n2 < fp.params.size(); ++n2){
                narray = {n1, n2};
                if (fabs(fp.params[n1][1] - fp.params[n2][1]) < cutoff){
                    active_clusters[narray] = true;
                }
                else {
                    active_clusters[narray] = false;
                }
                if (fp.maxp > 2){
                    for (int n3 = 0; n3 < fp.params.size(); ++n3){
                        narray = {n1, n2, n3};
                        if (fabs(fp.params[n1][1] - fp.params[n3][1]) < cutoff
                                and fabs(fp.params[n2][1] - fp.params[n3][1]) < cutoff){
                            active_clusters[narray] = true;
                        }
                        else {
                            active_clusters[narray] = false;
                        }
                    }
                }
            }
        }
    }
}
*/

void ModelParams::find_active_clusters(const feature_params& fp){

    if (fp.maxp > 1){
        //double cutoff = 0.0;
        //for (const auto& p: fp.params) cutoff += p[0];
        //cutoff *= 2.5 / fp.params.size();

        double cutoff = fp.cutoff / 3.0 - 0.1;

        vector1i narray;
        for (int n1 = 0; n1 < fp.params.size(); ++n1){
            for (int n2 = 0; n2 < fp.params.size(); ++n2){
                narray = {n1, n2};
                if ((fp.params[n1][1] > cutoff) and (fp.params[n2][1] > cutoff)){
                    active_clusters[narray] = false;
                }
                else {
                    active_clusters[narray] = true;
                }
                if (fp.maxp > 2){
                    for (int n3 = 0; n3 < fp.params.size(); ++n3){
                        narray = {n1, n2, n3};
                        if (
                            (fp.params[n1][1] > cutoff)
                            and (fp.params[n2][1] > cutoff)
                            and (fp.params[n3][1] > cutoff)
                        ){
                            active_clusters[narray] = false;
                        }
                        else {
                            active_clusters[narray] = true;
                        }
                    }
                }
            }
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
        const auto& type1_1 = linear_terms[iarray[i1]].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[iarray[i2]].type1;
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
        const auto& type1_1 = linear_terms[iarray[i1]].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[iarray[i2]].type1;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                const auto& type1_3 = linear_terms[iarray[i3]].type1;
                type_array = {type1_1, type1_2, type1_3};
                intersection = intersection_types_in_polynomial(type_array);
                if (intersection.size() > 0){
                    comb3.push_back(vector1i({iarray[i3], iarray[i2], iarray[i1]}));
                    for (const auto& type: intersection){
                        comb3_indices[type].emplace_back(i_comb);
                    }
                    ++i_comb;
                }
            }
        }
    }
}

void ModelParams::combination2_gtinv_cutoff(const vector1i& iarray){

    vector2i type_array;
    vector1i intersection;
    int i_comb(0);
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        const auto& type1_1 = linear_terms[iarray[i1]].type1;
        const auto& n1 = linear_terms[iarray[i1]].n;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[iarray[i2]].type1;
            const auto& n2 = linear_terms[iarray[i2]].n;
            if (active_clusters[vector1i({n1, n2})] == true){
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
}


void ModelParams::combination3_gtinv_cutoff(const vector1i& iarray){

    vector2i type_array;
    vector1i intersection;
    int i_comb(0);
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        const auto& type1_1 = linear_terms[iarray[i1]].type1;
        const auto& n1 = linear_terms[iarray[i1]].n;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[iarray[i2]].type1;
            const auto& n2 = linear_terms[iarray[i2]].n;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                const auto& type1_3 = linear_terms[iarray[i3]].type1;
                const auto& n3 = linear_terms[iarray[i3]].n;
                if (active_clusters[vector1i({n1, n2, n3})] == true){
                    type_array = {type1_1, type1_2, type1_3};
                    intersection = intersection_types_in_polynomial(type_array);
                    if (intersection.size() > 0){
                        comb3.push_back(vector1i({iarray[i3], iarray[i2], iarray[i1]}));
                        for (const auto& type: intersection){
                            comb3_indices[type].emplace_back(i_comb);
                        }
                        ++i_comb;
                    }
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

void ModelParams::combination1(const Mapping& mapping){

    const auto& ntp_attrs = mapping.get_ntp_attrs();
    int i(0);
    for (const auto& ntp: ntp_attrs){
        for (int type1 = 0; type1 < n_type; ++type1){
            if (check_type_in_type_pairs(vector1i({ntp.tp}), type1) == true){
                comb1_indices[type1].emplace_back(i);
            }
        }
        ++i;
    }
}


void ModelParams::combination2(const vector1i& iarray, const Mapping& mapping){

    const auto& ntp_attrs = mapping.get_ntp_attrs();
    int i_comb(0), tp1, tp2;
    bool match;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        tp1 = ntp_attrs[iarray[i1]].tp;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            tp2 = ntp_attrs[iarray[i2]].tp;
            match = false;
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_in_type_pairs(vector1i({tp1, tp2}), type1) == true){
                    comb2_indices[type1].emplace_back(i_comb);
                    match = true;
                }
            }
            if (match == true) {
                comb2.push_back(vector1i({iarray[i2], iarray[i1]}));
                ++i_comb;
            }
        }
    }

}

void ModelParams::combination3(const vector1i& iarray, const Mapping& mapping){

    const auto& ntp_attrs = mapping.get_ntp_attrs();
    int i_comb(0), tp1, tp2, tp3;
    bool match;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        tp1 = ntp_attrs[iarray[i1]].tp;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            tp2 = ntp_attrs[iarray[i2]].tp;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                tp3 = ntp_attrs[iarray[i3]].tp;
                match = false;
                for (int type1 = 0; type1 < n_type; ++type1){
                    if (check_type_in_type_pairs(vector1i({tp1, tp2, tp3}), type1) == true){
                        comb3_indices[type1].emplace_back(i_comb);
                        match = true;
                    }
                }
                if (match == true) {
                    comb3.push_back(vector1i({iarray[i3], iarray[i2], iarray[i1]}));
                    ++i_comb;
                }
            }
        }
    }
}

void ModelParams::combination2_cutoff(const vector1i& iarray, const Mapping& mapping){

    const auto& ntp_attrs = mapping.get_ntp_attrs();
    int i_comb(0), tp1, tp2;
    bool match;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        tp1 = ntp_attrs[iarray[i1]].tp;
        const auto& n1 = ntp_attrs[iarray[i1]].n;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            tp2 = ntp_attrs[iarray[i2]].tp;
            const auto& n2 = ntp_attrs[iarray[i2]].n;
            match = false;
            if (active_clusters[vector1i({n1, n2})] == true){
                for (int type1 = 0; type1 < n_type; ++type1){
                    if (check_type_in_type_pairs(vector1i({tp1, tp2}), type1) == true){
                        comb2_indices[type1].emplace_back(i_comb);
                        match = true;
                    }
                }
            }
            if (match == true) {
                comb2.push_back(vector1i({iarray[i2], iarray[i1]}));
                ++i_comb;
            }
        }
    }

}

void ModelParams::combination3_cutoff(const vector1i& iarray, const Mapping& mapping){

    const auto& ntp_attrs = mapping.get_ntp_attrs();
    int i_comb(0), tp1, tp2, tp3;
    bool match;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        tp1 = ntp_attrs[iarray[i1]].tp;
        const auto& n1 = ntp_attrs[iarray[i1]].n;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            tp2 = ntp_attrs[iarray[i2]].tp;
            const auto& n2 = ntp_attrs[iarray[i2]].n;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                tp3 = ntp_attrs[iarray[i3]].tp;
                const auto& n3 = ntp_attrs[iarray[i3]].n;
                match = false;
                if (active_clusters[vector1i({n1, n2, n3})] == true){
                    for (int type1 = 0; type1 < n_type; ++type1){
                        if (check_type_in_type_pairs(vector1i({tp1, tp2, tp3}), type1) == true){
                            comb3_indices[type1].emplace_back(i_comb);
                            match = true;
                        }
                    }
                }
                if (match == true) {
                    comb3.push_back(vector1i({iarray[i3], iarray[i2], iarray[i1]}));
                    ++i_comb;
                }
            }
        }
    }
}

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

const std::vector<struct LinearTerm>& ModelParams::get_linear_terms() const{
    return linear_terms;
}
const vector3i& ModelParams::get_tp_combs() const{
    return tp_combs;
}
