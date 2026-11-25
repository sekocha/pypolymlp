/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MODEL_PARAMS_POLYNOMIAL
#define __POLYMLP_MODEL_PARAMS_POLYNOMIAL

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_model_params_gtinv.h"


class ModelParamsPoly {

    int n_type;
    vector2i type_pairs;

    int n_linear_features;
    vector1i polynomial_indices;
    vector2i comb, comb2, comb3, comb1_indices, comb2_indices, comb3_indices;


    void select_linear_terms_pair(const feature_params& fp);

    void select_linear_terms_gtinv(
        const feature_params& fp,
        const std::vector<LinearTerm>& linear_terms
    );

    void set_polynomials_pair(const feature_params& fp, const Maps& maps);

    void set_polynomials_gtinv(
        const feature_params& fp,
        const Maps& maps,
        const std::vector<LinearTerm>& linear_terms
    );

    void combination1_pair(const Maps& maps);
    void combination2_pair(const vector1i& iarray, const Maps& maps);
    void combination3_pair(const vector1i& iarray, const Maps& maps);

    void combination1_gtinv(const std::vector<LinearTerm>& linear_terms);
    void combination2_gtinv(
        const vector1i& iarray, const std::vector<LinearTerm>& linear_terms
    );
    void combination3_gtinv(
        const vector1i& iarray, const std::vector<LinearTerm>& linear_terms
    );

    void append_combs_pair(
        const vector1i& tp_array,
        const vector1i& comb,
        vector2i& target_comb,
        vector2i& target_comb_indices,
        int& i_comb
    );

    void append_combs_gtinv(
        const vector2i& type_array,
        const vector1i& comb,
        vector2i& target_comb,
        vector2i& target_comb_indices,
        int& i_comb
    );

    vector1i intersection_types_in_polynomial(const vector2i &type1_array);

    public:

    ModelParamsPoly();
    ModelParamsPoly(
        const feature_params& fp,
        const Maps& maps,
        const std::vector<LinearTerm>& linear_terms
    );
    ~ModelParamsPoly();

    const vector2i& get_comb2() const;
    const vector2i& get_comb3() const;
    const vector1i& get_comb1_indices(const int type) const;
    const vector1i& get_comb2_indices(const int type) const;
    const vector1i& get_comb3_indices(const int type) const;

    const int get_n_linear_features() const;

};

#endif
