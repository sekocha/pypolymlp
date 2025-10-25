/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_API
#define __POLYMLP_API

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_parse_potential.h"
#include "polymlp_parse_potential_legacy.h"

#include "polymlp_mapping.h"
#include "polymlp_model_params.h"
#include "polymlp_features.h"
#include "polymlp_potential.h"


class PolymlpAPI {

    feature_params fp;
    vector1d pot;
    int n_variables;
    bool use_features, use_potential, use_model_params;

    Features features;
    Potential pmodel;
    Mapping mapping;
    ModelParams modelp;

    public:

    PolymlpAPI();
    ~PolymlpAPI();

    int parse_polymlp_file(
        const char *file,
        std::vector<std::string>& ele,
        vector1d& mass
    );
    int set_potential_model(const feature_params& fp, const vector1d& pot);
    int set_features(const feature_params& fp);
    int set_model_parameters(const feature_params& fp);

    int compute_anlmtp_conjugate(
        const vector1d& anlmtp_r,
        const vector1d& anlmtp_i,
        const int type1,
        vector1dc& anlmtp
    );

    int compute_anlmtp_conjugate(
        const vector2d& anlmtp_r,
        const vector2d& anlmtp_i,
        const int type1,
        vector2dc& anlmtp
    );

    int compute_features(
        const vector1d& antp,
        const int type1,
        vector1d& feature_values
    );

    int compute_features(
        const vector1dc& anlmtp,
        const int type1,
        vector1d& feature_values
    );

    int compute_features_deriv(
        const vector1dc& anlmtp,
        const vector2dc& anlmtp_dfx,
        const vector2dc& anlmtp_dfy,
        const vector2dc& anlmtp_dfz,
        const vector2dc& anlmtp_ds,
        const int type1,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
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

    const feature_params& get_fp() const;
    Maps& get_maps();
    const ModelParams& get_model_params() const;
    int get_n_variables();
};

#endif
