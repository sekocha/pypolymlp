/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_POTENTIAL
#define __POLYMLP_POTENTIAL

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_features.h"


class Potential {

    int n_type;
    Features f_obj;
    bool elim_conj;

    vector3i prod_features;
    std::vector<std::vector<PotentialModel> > potential_model;

    int set_mapping_prod_of_features();
    int set_terms_using_mapping(const vector1d& pot);
    void sort_potential_model();
    void release_memory();

    public:

    Potential();
    Potential(const feature_params& fp, const vector1d& pot);
    ~Potential();

    Maps& get_maps();

    void compute_features(
        const vector1d& antp,
        const int type1,
        vector1d& feature_values
    );

    void compute_features(
        const vector1dc& anlmtp,
        const int type1,
        vector1d& feature_values
    );

    void compute_prod_antp_deriv(
        const vector1d& antp,
        const int type1,
        vector1d& prod_antp_deriv
    );

    void compute_prod_anlmtp_deriv(
        const vector1dc& anlmtp,
        const int type1,
        vector1dc& prod_anlmtp_deriv
    );

    void compute_prod_features(
        const vector1d& features,
        const int type1,
        vector1d& prod_features
    );

    void compute_sum_of_prod_antp(
        const vector1d& antp,
        const int type1,
        vector1d& prod_sum_e,
        vector1d& prod_sum_f
    );

    void compute_sum_of_prod_anlmtp(
        const vector1dc& anlmtp,
        const int type1,
        vector1dc& prod_sum_e,
        vector1dc& prod_sum_f
    );

};

#endif
