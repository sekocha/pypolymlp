/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_API
#define __POLYMLP_API

#include "polymlp_mlpcpp.h"
#include "polymlp_eval.h"
#include "polymlp_parse_potential.h"
#include "polymlp_parse_potential_legacy.h"

#include "polymlp_features.h"
#include "polymlp_potential.h"


class PolymlpAPI {

    feature_params fp;
    vector1d pot;

    Features features;
    Potential pmodel;

    public:

    PolymlpAPI();
    ~PolymlpAPI();

    int parse_polymlp_file(
        const char *file,
        std::vector<std::string>& ele,
        vector1d& mass
    );

    int compute_anlmtp_conjugate(
        const vector1d& anlmtp_r,
        const vector1d& anlmtp_i,
        const int type1,
        vector1dc& anlmtp
    );

    int compute_sum_of_prod_antp(
        const vector1d& antp,
        const int type1,
        vector1d& prod_sum_e,
        vector1d& prod_sum_f
    );

    int compute_sum_of_prod_anlmtp(
        const vector1dc& anlmtp,
        const int type1,
        vector1dc& prod_sum_e,
        vector1dc& prod_sum_f
    );


    int set_features(const feature_params& fp);
    int set_potential_model();
    // int compute_features();

    const feature_params& get_fp() const;
    Maps& get_maps();
};

#endif
