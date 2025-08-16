/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params_polynomial.h"


ModelParamsPoly::ModelParamsPoly(){}
ModelParamsPoly::ModelParamsPoly(
    const feature_params& fp,
    const Maps& maps,
    const std::vector<LinearTerm>& linear_terms
){

    n_type = fp.n_type;
    type_pairs = maps.type_pairs;

    comb1_indices.resize(n_type);
    comb2_indices.resize(n_type);
    comb3_indices.resize(n_type);

    if (fp.feature_type == "pair") {
        n_linear_features = maps.ntp_attrs.size();
        select_linear_terms_pair(fp);
        set_polynomials_pair(fp, maps);
    }
    else if (fp.feature_type == "gtinv") {
        n_linear_features = linear_terms.size();
        select_linear_terms_gtinv(fp, linear_terms);
        set_polynomials_gtinv(fp, maps, linear_terms);
    }
}

ModelParamsPoly::~ModelParamsPoly(){}


void ModelParamsPoly::select_linear_terms_pair(const feature_params& fp){

    if (fp.model_type > 2){
        std::cerr << "Polymlp: Model type error." << std::endl;
        exit(8);
    }
    if (fp.maxp > 3){
        std::cerr << "Polymlp: maxp must be smaller than or equal to 3." << std::endl;
        exit(8);
    }
    if (fp.model_type == 2){
        for (int n = 0; n < n_linear_features; ++n){
            polynomial_indices.emplace_back(n);
        }
    }
}


void ModelParamsPoly::select_linear_terms_gtinv(
    const feature_params& fp,
    const std::vector<LinearTerm>& linear_terms
){

    if (fp.model_type > 4){
        std::cerr << "Polymlp: Model type error." << std::endl;
        exit(8);
    }
    if (fp.maxp > 3){
        std::cerr << "Polymlp: maxp must be smaller than or equal to 3." << std::endl;
        exit(8);
    }

    if (fp.model_type == 2){
        for (int n = 0; n < n_linear_features; ++n){
            polynomial_indices.emplace_back(n);
        }
    }
    else if (fp.model_type > 2){
        int max_order;
        if (fp.model_type == 3) max_order = 1;
        if (fp.model_type == 4) max_order = 2;
        for (int n = 0; n < n_linear_features; ++n){
            if (linear_terms[n].order <= max_order)
                polynomial_indices.emplace_back(n);
        }
    }
}


void ModelParamsPoly::set_polynomials_pair(const feature_params& fp, const Maps& maps){

    combination1_pair(maps);
    if (fp.model_type > 1){
        if (fp.maxp > 1) combination2_pair(polynomial_indices, maps);
        if (fp.maxp > 2) combination3_pair(polynomial_indices, maps);
    }
}


void ModelParamsPoly::set_polynomials_gtinv(
    const feature_params& fp,
    const Maps& maps,
    const std::vector<LinearTerm>& linear_terms
){

    combination1_gtinv(linear_terms);
    if (fp.model_type > 1){
        if (fp.maxp > 1) combination2_gtinv(polynomial_indices, linear_terms);
        if (fp.maxp > 2) combination3_gtinv(polynomial_indices, linear_terms);
    }
}

void ModelParamsPoly::combination1_pair(const Maps& maps){

    int i(0);
    vector1i tp_array;
    for (const auto& ntp: maps.ntp_attrs){
        tp_array = {ntp.tp};
        for (int type1 = 0; type1 < n_type; ++type1){
            if (check_type_in_type_pairs(tp_array, type_pairs, type1) == true){
                comb1_indices[type1].emplace_back(i);
            }
        }
        ++i;
    }
}


void ModelParamsPoly::combination2_pair(const vector1i& iarray, const Maps& maps){

    int i_comb(0), tp1, tp2;
    vector1i tp_array, comb;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        tp1 = maps.ntp_attrs[iarray[i1]].tp;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            tp2 = maps.ntp_attrs[iarray[i2]].tp;
            tp_array = {tp1, tp2};
            comb = {iarray[i2], iarray[i1]};
            append_combs_pair(tp_array, comb, comb2, comb2_indices, i_comb);
        }
    }
}


void ModelParamsPoly::combination3_pair(const vector1i& iarray, const Maps& maps){

    int i_comb(0), tp1, tp2, tp3;
    vector1i tp_array, comb;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        tp1 = maps.ntp_attrs[iarray[i1]].tp;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            tp2 = maps.ntp_attrs[iarray[i2]].tp;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                tp3 = maps.ntp_attrs[iarray[i3]].tp;
                tp_array = {tp1, tp2, tp3};
                comb = {iarray[i3], iarray[i2], iarray[i1]};
                append_combs_pair(tp_array, comb, comb3, comb3_indices, i_comb);
            }
        }
    }
}


void ModelParamsPoly::append_combs_pair(
    const vector1i& tp_array,
    const vector1i& comb,
    vector2i& target_comb,
    vector2i& target_comb_indices,
    int& i_comb
){

    bool match = false;
    for (int type1 = 0; type1 < n_type; ++type1){
        if (check_type_in_type_pairs(tp_array, type_pairs, type1) == true){
            target_comb_indices[type1].emplace_back(i_comb);
            match = true;
        }
    }
    if (match == true) {
        target_comb.push_back(comb);
        ++i_comb;
    }
}


void ModelParamsPoly::combination1_gtinv(const std::vector<LinearTerm>& linear_terms){

    for (size_t i_comb = 0; i_comb < linear_terms.size(); ++i_comb){
        for (const auto& type: linear_terms[i_comb].type1){
            comb1_indices[type].emplace_back(int(i_comb));
        }
    }
}


void ModelParamsPoly::combination2_gtinv(
    const vector1i& iarray, const std::vector<LinearTerm>& linear_terms
){

    vector2i type_array;
    vector1i comb;
    int i_comb(0);
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        const auto& type1_1 = linear_terms[iarray[i1]].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[iarray[i2]].type1;
            type_array = {type1_1, type1_2};

            comb = {iarray[i2], iarray[i1]};
            append_combs_gtinv(type_array, comb, comb2, comb2_indices, i_comb);
        }
    }
}

void ModelParamsPoly::combination3_gtinv(
    const vector1i& iarray, const std::vector<LinearTerm>& linear_terms
){

    vector2i type_array;
    vector1i comb;
    int i_comb(0);
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        const auto& type1_1 = linear_terms[iarray[i1]].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            const auto& type1_2 = linear_terms[iarray[i2]].type1;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                const auto& type1_3 = linear_terms[iarray[i3]].type1;
                type_array = {type1_1, type1_2, type1_3};
                comb = {iarray[i3], iarray[i2], iarray[i1]};
                append_combs_gtinv(type_array, comb, comb3, comb3_indices, i_comb);
            }
        }
    }
}


void ModelParamsPoly::append_combs_gtinv(
    const vector2i& type_array,
    const vector1i& comb,
    vector2i& target_comb,
    vector2i& target_comb_indices,
    int& i_comb
){
    vector1i intersection = intersection_types_in_polynomial(type_array);
    if (intersection.size() > 0){
        target_comb.emplace_back(comb);
        for (const auto& type: intersection){
            target_comb_indices[type].emplace_back(i_comb);
        }
        ++i_comb;
    }
}


vector1i ModelParamsPoly::intersection_types_in_polynomial(const vector2i &type1_array){

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


const vector2i& ModelParamsPoly::get_comb2() const { return comb2; }
const vector2i& ModelParamsPoly::get_comb3() const { return comb3; }
const vector1i& ModelParamsPoly::get_comb1_indices(const int type) const {
    return comb1_indices[type];
}
const vector1i& ModelParamsPoly::get_comb2_indices(const int type) const {
    return comb2_indices[type];
}
const vector1i& ModelParamsPoly::get_comb3_indices(const int type) const {
    return comb3_indices[type];
}
const int ModelParamsPoly::get_n_linear_features() const { return n_linear_features; }
