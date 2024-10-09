/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MODEL_PARAMS
#define __POLYMLP_MODEL_PARAMS

#include "polymlp_mlpcpp.h"
#include "polymlp_mapping.h"


template < typename SEQUENCE >
void Permutenr
(const SEQUENCE& input, SEQUENCE output,
 std::vector<SEQUENCE>& all, std::size_t r){
    if( output.size() == r ) all.emplace_back(output);
    else {
        for( std::size_t i=0; i < input.size(); ++i ) {
            SEQUENCE temp_output = output;
            temp_output.push_back(input[i]);
            Permutenr(input, temp_output, all, r);
        }
    }
}

struct LinearTermGtinv {
    int lmindex;
    vector1i tcomb_index;
    vector1i type1;
};

struct LinearTerm {
    int n;
    int n_id;
    int lm_comb_id;
    int tp_comb_id;
    int order;
    vector1i type1;
};


class ModelParams{

    int n_type, n_fn, n_linear_features, n_type_pairs, n_coeff_all;
    vector2i comb, comb2, comb3, comb1_indices, comb2_indices, comb3_indices;

    vector2i type_pairs;
    vector3i tp_combs, params_conditional;

    std::vector<struct LinearTermGtinv> linear_array_g;
    std::vector<struct LinearTerm> linear_terms;

    void initial_setting(const feature_params& fp, const Mapping& mapping);
    void enumerate_tp_combs(const feature_params& fp, const Mapping& mapping);
    void uniq_gtinv_type(const feature_params& fp, const Mapping& mapping);


    void combination1();
    void combination2(const vector1i& iarray);
    void combination3(const vector1i& iarray);
    void combination1_gtinv();
    void combination2_gtinv(const vector1i& iarray);
    void combination3_gtinv(const vector1i& iarray);

    vector1i vector_intersection(vector1i v1, vector1i v2);
    vector1i intersection_types_in_polynomial(const vector2i &type1_array);
    bool check_type_pairs(const vector1i& index, const int& type1) const;
    int find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb);

    int seq2typecomb(const int& seq);

    public:

    ModelParams();
    ModelParams(const feature_params& fp, const Mapping& mapping);
    ~ModelParams();

//    const int& get_n_type() const;
//    const int& get_n_type_pairs() const;
//    const int& get_n_fn() const;
    const int& get_n_linear_features() const;
    const int& get_n_coeff_all() const;

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;

    const vector1i& get_comb1_indices(const int type) const;
    const vector1i& get_comb2_indices(const int type) const;
    const vector1i& get_comb3_indices(const int type) const;

//    const vector2i& get_type_pairs() const;
//    const vector2i& get_type_pair_to_nlist() const;

    const std::vector<struct LinearTermGtinv>& get_linear_term_gtinv() const;
    const std::vector<struct LinearTerm>& get_linear_terms() const;

    const vector3i& get_tp_combs() const;
};

#endif
