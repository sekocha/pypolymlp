/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MODEL_PARAMS_GTINV
#define __POLYMLP_MODEL_PARAMS_GTINV

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"


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


void _enumerate_tp_combs(Maps& maps, const int gtinv_order, vector3i& tp_combs);

void _enumerate_nonzero_n(
    Maps& maps, const vector3i& tp_combs, vector3i& nonzero_n_list
);

int _find_tp_comb_id(const vector2i& tp_comb_ref, const vector1i& tp_comb);

bool check_type_in_type_pairs(
    const vector1i& tp_comb, const vector2i& type_pairs, const int& type1
);

vector1i vector_intersection(vector1i v1, vector1i v2);

void uniq_gtinv_type(
    const feature_params& fp,
    Maps& maps,
    vector3i& tp_combs,
    std::vector<LinearTerm>& linear_terms
);

#endif
