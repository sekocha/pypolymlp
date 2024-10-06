/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MODEL_PARAMS
#define __POLYMLP_MODEL_PARAMS

#include "polymlp_mlpcpp.h"


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


class ModelParams{

    int n_type, n_fn, n_des, n_coeff_all, n_type_pairs;
    vector2i comb, comb2, comb3, comb1_indices, comb2_indices, comb3_indices;

    vector2i type_pairs;
    vector2i tp_nlist_map;
    vector3i tp_combs;

    std::vector<struct LinearTermGtinv> linear_array_g;

    void set_type_pairs(const feature_params& fp);
    void set_type_pairs_charge(const feature_params& fp);
    void enumerate_tp_combs(const feature_params& fp);

    void combination1();
    void combination2(const vector1i& iarray);
    void combination3(const vector1i& iarray);
    void combination1_gtinv();
    void combination2_gtinv(const vector1i& iarray);
    void combination3_gtinv(const vector1i& iarray);

    int seq2typecomb(const int& i);
    int seq2igtinv(const int& seq);

    bool check_type_pairs(const vector1i& index, const int& type1) const;
    void uniq_gtinv_type(const feature_params& fp);

    vector1i intersection_types_in_polynomial(const vector2i &type1_array);

    void initial_setting(const struct feature_params& fp);

    int find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb);

    public:

    ModelParams();
    ModelParams(const struct feature_params& fp);
    ModelParams(const struct feature_params& fp, const bool icharge);
    ~ModelParams();

    const int& get_n_type() const;
    const int& get_n_type_pairs() const;
    const int& get_n_fn() const;
    const int& get_n_des() const;
    const int& get_n_coeff_all() const;

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;

    const vector1i& get_comb1_indices(const int type) const;
    const vector1i& get_comb2_indices(const int type) const;
    const vector1i& get_comb3_indices(const int type) const;

    const vector2i& get_type_pairs() const;
    const std::vector<struct LinearTermGtinv>& get_linear_term_gtinv() const;

};

#endif
