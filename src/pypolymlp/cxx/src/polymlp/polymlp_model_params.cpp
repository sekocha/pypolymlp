/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_model_params.h"

ModelParams::ModelParams(){}
ModelParams::ModelParams(const struct feature_params& fp){

    n_type = fp.n_type;
    set_type_pairs();
    initial_setting(fp);
}

ModelParams::ModelParams(const struct feature_params& fp, const bool icharge){

    n_type = fp.n_type;
    if (icharge == false) set_type_pairs();
    else set_type_pairs_charge();
    initial_setting(fp);
}

ModelParams::~ModelParams(){}

void ModelParams::set_type_pairs(){
    // Setting two elements in one array such as {0,1}, ... is unavailable.
    // If using such a setting, codes for computing features should be revised.
    type_pairs.resize(n_type);
    for (auto& tp1: type_pairs) tp1.resize(n_type);

    if (n_type == 1) {
        n_type_pairs = 1;
        type_pairs[0][0] = 0;
    }
    else if (n_type == 2) {
        n_type_pairs = 3;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = type_pairs[1][0] = 1;
        type_pairs[1][1] = 2;
    }
    else if (n_type == 3) {
        n_type_pairs = 6;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = type_pairs[1][0] = 1;
        type_pairs[0][2] = type_pairs[2][0] = 2;
        type_pairs[1][1] = 3;
        type_pairs[1][2] = type_pairs[2][1] = 4;
        type_pairs[2][2] = 5;
    }
    else if (n_type == 4) {
        n_type_pairs = 10;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = type_pairs[1][0] = 1;
        type_pairs[0][2] = type_pairs[2][0] = 2;
        type_pairs[0][3] = type_pairs[3][0] = 3;
        type_pairs[1][1] = 4;
        type_pairs[1][2] = type_pairs[2][1] = 5;
        type_pairs[1][3] = type_pairs[3][1] = 6;
        type_pairs[2][2] = 7;
        type_pairs[2][3] = type_pairs[3][2] = 8;
        type_pairs[3][3] = 9;
    }
    else if (n_type == 5) {
        n_type_pairs = 15;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = type_pairs[1][0] = 1;
        type_pairs[0][2] = type_pairs[2][0] = 2;
        type_pairs[0][3] = type_pairs[3][0] = 3;
        type_pairs[0][4] = type_pairs[4][0] = 4;
        type_pairs[1][1] = 5;
        type_pairs[1][2] = type_pairs[2][1] = 6;
        type_pairs[1][3] = type_pairs[3][1] = 7;
        type_pairs[1][4] = type_pairs[4][1] = 8;
        type_pairs[2][2] = 9;
        type_pairs[2][3] = type_pairs[3][2] = 10;
        type_pairs[2][4] = type_pairs[4][2] = 11;
        type_pairs[3][3] = 12;
        type_pairs[3][4] = type_pairs[4][3] = 13;
        type_pairs[4][4] = 14;
    }
    else if (n_type == 6) {
        n_type_pairs = 21;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = type_pairs[1][0] = 1;
        type_pairs[0][2] = type_pairs[2][0] = 2;
        type_pairs[0][3] = type_pairs[3][0] = 3;
        type_pairs[0][4] = type_pairs[4][0] = 4;
        type_pairs[0][5] = type_pairs[5][0] = 5;
        type_pairs[1][1] = 6;
        type_pairs[1][2] = type_pairs[2][1] = 7;
        type_pairs[1][3] = type_pairs[3][1] = 8;
        type_pairs[1][4] = type_pairs[4][1] = 9;
        type_pairs[1][5] = type_pairs[5][1] = 10;
        type_pairs[2][2] = 11;
        type_pairs[2][3] = type_pairs[3][2] = 12;
        type_pairs[2][4] = type_pairs[4][2] = 13;
        type_pairs[2][5] = type_pairs[5][2] = 14;
        type_pairs[3][3] = 15;
        type_pairs[3][4] = type_pairs[4][3] = 16;
        type_pairs[3][5] = type_pairs[5][3] = 17;
        type_pairs[4][4] = 18;
        type_pairs[4][5] = type_pairs[5][4] = 19;
        type_pairs[5][5] = 20;
    }
    else if (n_type == 7) {
        n_type_pairs = 28;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = type_pairs[1][0] = 1;
        type_pairs[0][2] = type_pairs[2][0] = 2;
        type_pairs[0][3] = type_pairs[3][0] = 3;
        type_pairs[0][4] = type_pairs[4][0] = 4;
        type_pairs[0][5] = type_pairs[5][0] = 5;
        type_pairs[0][6] = type_pairs[6][0] = 6;
        type_pairs[1][1] = 7;
        type_pairs[1][2] = type_pairs[2][1] = 8;
        type_pairs[1][3] = type_pairs[3][1] = 9;
        type_pairs[1][4] = type_pairs[4][1] = 10;
        type_pairs[1][5] = type_pairs[5][1] = 11;
        type_pairs[1][6] = type_pairs[6][1] = 12;
        type_pairs[2][2] = 13;
        type_pairs[2][3] = type_pairs[3][2] = 14;
        type_pairs[2][4] = type_pairs[4][2] = 15;
        type_pairs[2][5] = type_pairs[5][2] = 16;
        type_pairs[2][6] = type_pairs[6][2] = 17;
        type_pairs[3][3] = 18;
        type_pairs[3][4] = type_pairs[4][3] = 19;
        type_pairs[3][5] = type_pairs[5][3] = 20;
        type_pairs[3][6] = type_pairs[6][3] = 21;
        type_pairs[4][4] = 22;
        type_pairs[4][5] = type_pairs[5][4] = 23;
        type_pairs[4][6] = type_pairs[6][4] = 24;
        type_pairs[5][5] = 25;
        type_pairs[5][6] = type_pairs[6][5] = 26;
        type_pairs[6][6] = 27;
    }
    else {
        std::cout << "Polymlp is now available for up to 7-ary systems." << std::endl;
        exit(8);
    }
}

void ModelParams::set_type_pairs_charge(){
    // Setting two elements in one array such as {0,1}, ... is unavailable.
    // If using such a setting, codes for computing features should be revised.
    type_pairs.resize(n_type);
    for (auto& tp1: type_pairs) tp1.resize(n_type);

    if (n_type == 2) {
        n_type_pairs = 4;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = 1;
        type_pairs[1][0] = 2;
        type_pairs[1][1] = 3;
    }
    else if (n_type == 3) {
        n_type_pairs = 9;
        type_pairs[0][0] = 0;
        type_pairs[0][1] = 1;
        type_pairs[0][2] = 2;
        type_pairs[1][0] = 3;
        type_pairs[1][1] = 4;
        type_pairs[1][2] = 5;
        type_pairs[2][0] = 6;
        type_pairs[2][1] = 7;
        type_pairs[2][2] = 8;
    }
    else {
        exit(8);
    }
}

void ModelParams::initial_setting(const struct feature_params& fp){

    n_type = fp.n_type, n_fn = fp.params.size();

    if (fp.des_type == "pair") n_des = n_fn * n_type_pairs;
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

void ModelParams::enumerate_tp_combs(const feature_params& fp, vector3i& tp_comb_candidates){

    const vector2i& l_comb = fp.l_comb;
    vector1i pinput(n_type_pairs);
    for (int i = 0; i < n_type_pairs; ++i) pinput[i] = i;

    const int gtinv_order = (*(l_comb.end() - 1)).size();
    tp_comb_candidates.resize(gtinv_order + 1);
    for (int order = 1; order <= gtinv_order; ++order){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, order);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_pairs(p1, type1) == true){
                    tp_comb_candidates[order].emplace_back(p1);
                    break;
                }
            }
        }
    }
}

void ModelParams::uniq_gtinv_type(const feature_params& fp){

    vector3i tp_comb_candidates;
    enumerate_tp_combs(fp, tp_comb_candidates);
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

    const vector2i& l_comb_all = fp.l_comb;
    for (size_t i = 0; i < l_comb_all.size(); ++i){
        const vector1i& l_comb = l_comb_all[i];
        const int order = l_comb.size();

        std::set<std::multiset<std::pair<int,int> > > uniq_lmt;
        for (const auto &tp_comb: tp_comb_candidates[order]){
            std::multiset<std::pair<int, int> > tmp;
            for (size_t j = 0; j < tp_comb.size(); ++j){
                tmp.insert(std::make_pair(l_comb[j], tp_comb[j]));
            }
            uniq_lmt.insert(tmp);
        }
        for (const auto& lt: uniq_lmt){
            vector1i tc, t1a;
            for (const auto& lt1: lt) tc.emplace_back(lt1.second);
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_pairs(tc, type1) == true)
                    t1a.emplace_back(type1);
            }
            linear_array_g.emplace_back(LinearTermGtinv({int(i),tc,t1a}));
        }
    }
}

/*
void ModelParams::uniq_gtinv_type(const feature_params& fp){

    const vector2i &l_comb = fp.l_comb;
    vector1i pinput(type_pairs.size());
    for (size_t i = 0; i < type_pairs.size(); ++i) pinput[i] = i;

    const int gtinv_order = (*(l_comb.end()-1)).size();
    vector3i perm_array(gtinv_order);
    for (int i = 0; i < gtinv_order; ++i){
        vector2i perm;
        Permutenr(pinput, vector1i({}), perm, i+1);
        for (const auto& p1: perm){
            for (int type1 = 0; type1 < n_type; ++type1){
                if (check_type_pairs(p1, type1) == true){
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
                if (check_type_pairs(tc, type1) == true)
                    t1a.emplace_back(type1);
            }
            linear_array_g.emplace_back(LinearTermGtinv({int(i),tc,t1a}));
        }
    }
}
*/

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
/*
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
*/


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
        if (std::find(tp_type1.begin(), tp_type1.end(), tp) == tp_type1.end()) return false;
    }
    return true;
}

/*
bool ModelParams::check_type_pairs(const vector1i& index, const int& type1) const{
    vector1i size;
    for (const auto& p2: index){
        size.emplace_back(type_pairs[p2][type1].size());
    }
    int minsize = *std::min_element(size.begin(), size.end());
    return minsize > 0;
}
*/

const int& ModelParams::get_n_type() const { return n_type; }
const int& ModelParams::get_n_type_pairs() const { return n_type_pairs; }
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

const vector2i& ModelParams::get_type_pairs() const{
    return type_pairs;
}

/*
vector1i ModelParams::get_type_pairs(const vector1i& tc_index, const int& type1){
    vector1i all;
    for (const auto& i: tc_index)
        all.emplace_back(type_pairs[i][type1][0]);
    return all;
}
*/

const std::vector<struct LinearTermGtinv>&
ModelParams::get_linear_term_gtinv() const{
    return linear_array_g;
}
