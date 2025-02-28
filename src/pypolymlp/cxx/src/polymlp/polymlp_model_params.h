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

struct LinearTerm {
    int n;
    int lm_comb_id;
    int tp_comb_id;
    int order;
    vector1i type1;
};


class ModelParams{

    int n_type, n_fn, n_linear_features, n_type_pairs, n_coeff_all;
    vector2i comb, comb2, comb3, comb1_indices, comb2_indices, comb3_indices;

    vector2i type_pairs;
    vector3i tp_combs, nonzero_n_list;

    std::vector<struct LinearTerm> linear_terms;
    std::unordered_map<vector1i,bool,HashVI> active_clusters;

    void initial_setting(const feature_params& fp, const Mapping& mapping);
    void polynomial_setting(const feature_params& fp, const Mapping& mapping);

    void enumerate_tp_combs(const int gtinv_order);
    void enumerate_nonzero_n(const Mapping& mapping);
    bool check_type_in_type_pairs(const vector1i& tp_comb, const int& type1) const;
    vector1i vector_intersection(vector1i v1, vector1i v2);
    int find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb);

    void uniq_gtinv_type(const feature_params& fp, const Mapping& mapping);

    void combination1(const Mapping& mapping);
    void combination2(const vector1i& iarray, const Mapping& mapping);
    void combination3(const vector1i& iarray, const Mapping& mapping);
    void combination1_gtinv();
    void combination2_gtinv(const vector1i& iarray);
    void combination3_gtinv(const vector1i& iarray);

    void find_active_clusters(const feature_params& fp);
    void combination2_cutoff(const vector1i& iarray, const Mapping& mapping);
    void combination3_cutoff(const vector1i& iarray, const Mapping& mapping);
    void combination2_gtinv_cutoff(const vector1i& iarray);
    void combination3_gtinv_cutoff(const vector1i& iarray);

    vector1i intersection_types_in_polynomial(const vector2i &type1_array);
    int seq2typecomb(const int& seq);

    public:

    ModelParams();
    ModelParams(const feature_params& fp, const Mapping& mapping);
    ~ModelParams();

    const int& get_n_linear_features() const;
    const int& get_n_coeff_all() const;

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;
    const vector1i& get_comb1_indices(const int type) const;
    const vector1i& get_comb2_indices(const int type) const;
    const vector1i& get_comb3_indices(const int type) const;

    const std::vector<struct LinearTerm>& get_linear_terms() const;
    const vector3i& get_tp_combs() const;
};

#endif
