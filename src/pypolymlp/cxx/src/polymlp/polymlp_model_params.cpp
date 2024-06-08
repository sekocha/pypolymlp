/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params.h"

ModelParams::ModelParams(){}
ModelParams::ModelParams(const struct feature_params& fp){

    n_type = fp.n_type;

    // Setting two elements in one array such as {0,1}, ... is unavailable.
    // If using such a setting, codes for computing features should be revised.

    if (n_type == 1) {
        type_comb_pair = {{{0}}};
    }
    else if (n_type == 2) {
        type_comb_pair = {{{0}, {}},
                          {{1}, {0}},
                          {{}, {1}}};
    }
    else if (n_type == 3) {
        type_comb_pair = {{{0}, {}, {}},
                          {{1}, {0}, {}},
                          {{2}, {}, {0}},
                          {{}, {1}, {}},
                          {{}, {2}, {1}},
                          {{}, {}, {2}}};
    }
    else if (n_type == 4) {
        type_comb_pair = {{{0}, {}, {}, {}},
                          {{1}, {0}, {}, {}},
                          {{2}, {}, {0}, {}},
                          {{3}, {}, {}, {0}},
                          {{}, {1}, {}, {}},
                          {{}, {2}, {1}, {}},
                          {{}, {3}, {}, {1}},
                          {{}, {}, {2}, {}},
                          {{}, {}, {3}, {2}},
                          {{}, {}, {}, {3}}};
    }
    else if (n_type == 5) {
        type_comb_pair = {{{0}, {}, {}, {}, {}},
                          {{1}, {0}, {}, {}, {}},
                          {{2}, {}, {0}, {}, {}},
                          {{3}, {}, {}, {0}, {}},
                          {{4}, {}, {}, {}, {0}},
                          {{}, {1}, {}, {}, {}},
                          {{}, {2}, {1}, {}, {}},
                          {{}, {3}, {}, {1}, {}},
                          {{}, {4}, {}, {}, {1}},
                          {{}, {}, {2}, {}, {}},
                          {{}, {}, {3}, {2}, {}},
                          {{}, {}, {4}, {}, {2}},
                          {{}, {}, {}, {3}, {}},
                          {{}, {}, {}, {4}, {3}},
                          {{}, {}, {}, {}, {4}}};
    }

    n_type_comb = type_comb_pair.size();
    initial_setting(fp);
}

ModelParams::ModelParams(const struct feature_params& fp, const bool icharge){

    n_type = fp.n_type;

    if (icharge == false){
        if (n_type == 1) {
            type_comb_pair = {{{0}}};
        }
        else if (n_type == 2) {
            type_comb_pair = {{{0}, {}},
                              {{1}, {0}},
                              {{}, {1}}};
        }
        else if (n_type == 3) {
            type_comb_pair = {{{0}, {}, {}},
                              {{1}, {0}, {}},
                              {{2}, {}, {0}},
                              {{}, {1}, {}},
                              {{}, {2}, {1}},
                              {{}, {}, {2}}};
        }
        else if (n_type == 4) {
            type_comb_pair = {{{0}, {}, {}, {}},
                              {{1}, {0}, {}, {}},
                              {{2}, {}, {0}, {}},
                              {{3}, {}, {}, {0}},
                              {{}, {1}, {}, {}},
                              {{}, {2}, {1}, {}},
                              {{}, {3}, {}, {1}},
                              {{}, {}, {2}, {}},
                              {{}, {}, {3}, {2}},
                              {{}, {}, {}, {3}}};
        }
        else if (n_type == 5) {
            type_comb_pair = {{{0}, {}, {}, {}, {}},
                              {{1}, {0}, {}, {}, {}},
                              {{2}, {}, {0}, {}, {}},
                              {{3}, {}, {}, {0}, {}},
                              {{4}, {}, {}, {}, {0}},
                              {{}, {1}, {}, {}, {}},
                              {{}, {2}, {1}, {}, {}},
                              {{}, {3}, {}, {1}, {}},
                              {{}, {4}, {}, {}, {1}},
                              {{}, {}, {2}, {}, {}},
                              {{}, {}, {3}, {2}, {}},
                              {{}, {}, {4}, {}, {2}},
                              {{}, {}, {}, {3}, {}},
                              {{}, {}, {}, {4}, {3}},
                              {{}, {}, {}, {}, {4}}};
        }
        else if (n_type == 6) {
            type_comb_pair = {{{0}, {}, {}, {}, {}, {}},
                              {{1}, {0}, {}, {}, {}, {}},
                              {{2}, {}, {0}, {}, {}, {}},
                              {{3}, {}, {}, {0}, {}, {}},
                              {{4}, {}, {}, {}, {0}, {}},
                              {{5}, {}, {}, {}, {}, {0}},
                              {{}, {1}, {}, {}, {}, {}},
                              {{}, {2}, {1}, {}, {}, {}},
                              {{}, {3}, {}, {1}, {}, {}},
                              {{}, {4}, {}, {}, {1}, {}},
                              {{}, {5}, {}, {}, {}, {1}},
                              {{}, {}, {2}, {}, {}, {}},
                              {{}, {}, {3}, {2}, {}, {}},
                              {{}, {}, {4}, {}, {2}, {}},
                              {{}, {}, {5}, {}, {}, {2}},
                              {{}, {}, {}, {3}, {}, {}},
                              {{}, {}, {}, {4}, {3}, {}},
                              {{}, {}, {}, {5}, {}, {3}},
                              {{}, {}, {}, {}, {4}, {}},
                              {{}, {}, {}, {}, {5}, {4}},
                              {{}, {}, {}, {}, {}, {5}}};
        }
        else if (n_type == 7) {
            type_comb_pair = {{{0}, {}, {}, {}, {}, {}, {}},
                              {{1}, {0}, {}, {}, {}, {}, {}},
                              {{2}, {}, {0}, {}, {}, {}, {}},
                              {{3}, {}, {}, {0}, {}, {}, {}},
                              {{4}, {}, {}, {}, {0}, {}, {}},
                              {{5}, {}, {}, {}, {}, {0}, {}},
                              {{6}, {}, {}, {}, {}, {}, {0}},
                              {{}, {1}, {}, {}, {}, {}, {}},
                              {{}, {2}, {1}, {}, {}, {}, {}},
                              {{}, {3}, {}, {1}, {}, {}, {}},
                              {{}, {4}, {}, {}, {1}, {}, {}},
                              {{}, {5}, {}, {}, {}, {1}, {}},
                              {{}, {6}, {}, {}, {}, {}, {1}},
                              {{}, {}, {2}, {}, {}, {}, {}},
                              {{}, {}, {3}, {2}, {}, {}, {}},
                              {{}, {}, {4}, {}, {2}, {}, {}},
                              {{}, {}, {5}, {}, {}, {2}, {}},
                              {{}, {}, {6}, {}, {}, {}, {2}},
                              {{}, {}, {}, {3}, {}, {}, {}},
                              {{}, {}, {}, {4}, {3}, {}, {}},
                              {{}, {}, {}, {5}, {}, {3}, {}},
                              {{}, {}, {}, {6}, {}, {}, {3}},
                              {{}, {}, {}, {}, {4}, {}, {}},
                              {{}, {}, {}, {}, {5}, {4}, {}},
                              {{}, {}, {}, {}, {6}, {}, {4}},
                              {{}, {}, {}, {}, {}, {5}, {}},
                              {{}, {}, {}, {}, {}, {6}, {5}},
                              {{}, {}, {}, {}, {}, {}, {6}}};
        }
        else {
            exit(8);
        }
    }
    else {
    // Setting two elements in one array such as {0,1}, ... is unavailable.
    // If using such a setting, codes for computing features should be revised.
        if (n_type == 2) {
            type_comb_pair = {{{0}, {}},
                              {{1}, {}},
                              {{}, {0}},
                              {{}, {1}}};
        }
        else if (n_type == 3) {
            type_comb_pair = {{{0}, {}, {}},
                              {{1}, {}, {}},
                              {{2}, {}, {}},
                              {{}, {0}, {}},
                              {{}, {1}, {}},
                              {{}, {2}, {}},
                              {{}, {}, {0}},
                              {{}, {}, {1}},
                              {{}, {}, {2}}};
        }
        else {
            exit(8);
        }
    }

    n_type_comb = type_comb_pair.size();
    initial_setting(fp);
}

ModelParams::~ModelParams(){}

void ModelParams::initial_setting(const struct feature_params& fp){

    n_type = fp.n_type, n_fn = fp.params.size();

    if (fp.des_type == "pair") n_des = n_fn * type_comb_pair.size();
    else if (fp.des_type == "gtinv"){
        uniq_gtinv_type(fp);
        n_des = n_fn * linear_array_g.size();
    }

    vector1i polynomial_index;
    if (fp.model_type == 2){
        for (int n = 0; n < n_des; ++n)
            polynomial_index.emplace_back(n);
    }
    else if (fp.model_type == 3 and fp.des_type == "gtinv"){
        for (size_t i = 0; i < linear_array_g.size(); ++i){
            const auto& lin = linear_array_g[i];
            if (lin.tcomb_index.size() == 1){
                for (int n = 0; n < n_fn; ++n){
                    polynomial_index.emplace_back(n*linear_array_g.size()+i);
                }
            }
        }
        std::sort(polynomial_index.begin(),polynomial_index.end());
    }
    else if (fp.model_type == 4 and fp.des_type == "gtinv"){
        for (size_t i = 0; i < linear_array_g.size(); ++i){
            const auto& lin = linear_array_g[i];
            if (lin.tcomb_index.size() < 3 ){
                for (int n = 0; n < n_fn; ++n){
                    polynomial_index.emplace_back(n*linear_array_g.size()+i);
                }
            }
        }
        std::sort(polynomial_index.begin(),polynomial_index.end());
    }

    comb1_indices.resize(n_type);
    comb2_indices.resize(n_type);
    comb3_indices.resize(n_type);

    if (fp.des_type == "pair") combination1();
    else if (fp.des_type == "gtinv") combination1_gtinv();

    if (fp.model_type == 1) n_coeff_all = n_des * fp.maxp;
    else if (fp.model_type > 1){
        if (fp.des_type == "pair"){
            if (fp.maxp > 1) combination2(polynomial_index);
            if (fp.maxp > 2) combination3(polynomial_index);
        }
        else if (fp.des_type == "gtinv"){
            if (fp.maxp > 1) combination2_gtinv(polynomial_index);
            if (fp.maxp > 2) combination3_gtinv(polynomial_index);
        }
        n_coeff_all = n_des + comb2.size() + comb3.size();
    }
}

void ModelParams::uniq_gtinv_type(const feature_params& fp){

    const vector2i &l_comb = fp.l_comb;
    vector1i pinput(type_comb_pair.size());
    for (size_t i = 0; i < type_comb_pair.size(); ++i) pinput[i] = i;

    const int gtinv_order = (*(l_comb.end()-1)).size();
    vector3i perm_array(gtinv_order);
    for (int i = 0; i < gtinv_order; ++i){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, i+1);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_comb_pair(p1, type1) == true){
                    perm_array[i].emplace_back(p1);
                    break;
                }
            }
        }
    }

    for (size_t i = 0; i < l_comb.size(); ++i){
        const vector1i& lc = l_comb[i];
        std::set<std::multiset<std::pair<int,int> > > uniq_lmt;
        for (const auto &p: perm_array[lc.size()-1]){
            std::multiset<std::pair<int, int> > tmp;
            for (size_t j = 0; j < p.size(); ++j){
                tmp.insert(std::make_pair(lc[j], p[j]));
            }
            uniq_lmt.insert(tmp);
        }
        for (const auto& lt: uniq_lmt){
            vector1i tc, t1a;
            for (const auto& lt1: lt) tc.emplace_back(lt1.second);
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_comb_pair(tc, type1) == true)
                    t1a.emplace_back(type1);
            }
            linear_array_g.emplace_back(LinearTermGtinv({int(i),tc,t1a}));
        }
    }
}

void ModelParams::combination1_gtinv(){

    int i_comb;
    for (int n = 0; n < n_fn; ++n){
        for (size_t i = 0; i < linear_array_g.size(); ++i){
            i_comb = n * linear_array_g.size() + i;
            for (const auto& type: linear_array_g[i].type1){
                comb1_indices[type].emplace_back(i_comb);
            }
        }
    }
}

void ModelParams::combination2_gtinv(const vector1i& iarray){

    vector2i type_array;
    vector1i intersection;
    int i_comb(0), t1, t2;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        t1 = seq2igtinv(iarray[i1]);
        const auto &type1_1 = linear_array_g[t1].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            t2 = seq2igtinv(iarray[i2]);
            const auto &type1_2 = linear_array_g[t2].type1;
            type_array = {type1_1, type1_2};
            intersection = intersection_types_in_polynomial(type_array);
            if (intersection.size() > 0){
                comb2.push_back(vector1i({iarray[i2],iarray[i1]}));
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
    int i_comb(0), t1, t2, t3;
    for (size_t i1 = 0; i1 < iarray.size(); ++i1){
        t1 = seq2igtinv(iarray[i1]);
        const auto &type1_1 = linear_array_g[t1].type1;
        for (size_t i2 = 0; i2 <= i1; ++i2){
            t2 = seq2igtinv(iarray[i2]);
            const auto &type1_2 = linear_array_g[t2].type1;
            for (size_t i3 = 0; i3 <= i2; ++i3){
                t3 = seq2igtinv(iarray[i3]);
                const auto &type1_3 = linear_array_g[t3].type1;
                type_array = {type1_1, type1_2, type1_3};
                intersection = intersection_types_in_polynomial(type_array);

                if (intersection.size() > 0){
                    comb3.push_back
                        (vector1i({iarray[i3],iarray[i2],iarray[i1]}));
                    for (const auto& type: intersection){
                        comb3_indices[type].emplace_back(i_comb);
                    }
                    ++i_comb;
                }
            }
        }
    }
}

bool ModelParams::check_type(const vector2i &type1_array){

    for (int type1 = 0; type1 < n_type; ++type1){
        bool tag = true;
        for (const auto &t1: type1_array){
            if (std::find(t1.begin(),t1.end(),type1) == t1.end()){
                tag = false;
                break;
            }
        }
        if (tag == true) return true;
    }
    return false;
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
int ModelParams::seq2igtinv(const int& seq){
    return seq % linear_array_g.size();
}

void ModelParams::combination1(){

    int t1;
    for (int i = 0; i < n_des; ++i){
        t1 = seq2typecomb(i);
        for (int type1 = 0; type1 < n_type; ++type1){
            if (check_type_comb_pair(vector1i({t1}), type1) == true){
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
                if (check_type_comb_pair(vector1i({t1,t2}), type1) == true){
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
                    if (check_type_comb_pair
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


bool ModelParams::check_type_comb_pair
(const vector1i& index, const int& type1) const{
    vector1i size;
    for (const auto& p2: index){
        size.emplace_back(type_comb_pair[p2][type1].size());
    }
    int minsize = *std::min_element(size.begin(), size.end());
    return minsize > 0;
}

const int& ModelParams::get_n_type() const { return n_type; }
const int& ModelParams::get_n_type_comb() const { return n_type_comb; }
const int& ModelParams::get_n_fn() const { return n_fn; }
const int& ModelParams::get_n_des() const { return n_des; }
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

const vector3i& ModelParams::get_type_comb_pair() const{
    return type_comb_pair;
}

vector1i ModelParams::get_type_comb_pair(const vector1i& tc_index,
                                         const int& type1){
    vector1i all;
    for (const auto& i: tc_index)
        all.emplace_back(type_comb_pair[i][type1][0]);
    return all;
}

const std::vector<struct LinearTermGtinv>&
ModelParams::get_linear_term_gtinv() const{
    return linear_array_g;
}
