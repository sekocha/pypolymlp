/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES
#define __POLYMLP_FEATURES

#include <map>

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_mapping.h"
#include "polymlp_model_params.h"
#include "polymlp_features_polynomial.h"
#include "polymlp_products.h"


class Features {

    int n_type;
    bool eliminate_conj;

    Mapping mapping;
    ModelParams modelp;
    FeaturesPoly poly;

    vector3i prod, prod_deriv;
    std::vector<MapFromVec> prod_map_deriv_from_keys;
    std::vector<MappedMultipleFeatures> mapped_features;

    int set_linear_features_pair();
    int set_linear_features_gtinv(const feature_params& fp);

    int set_mappings_standard();
    int set_mappings_efficient(const feature_params& fp);
    int set_deriv_mappings();
    int sort_mapped_features();

    int convert_to_mapped_features(
        const MultipleFeatures& features,
        const int t1,
        MapFromVec& prod_map_from_keys
    );

    int find_local_ids(
        Maps& maps,
        const int type1,
        const int n,
        const vector1i& lm_comb,
        const vector1i& tp_comb,
        vector1i& local_ids
    );

    public:

    Features();
    Features(const feature_params& fp, const bool eliminate_conj_i);
    ~Features();

    Maps& get_maps();
    MapFromVec& get_prod_map_deriv(const int type1);
    MapFromVec& get_prod_features_map(const int type1);

    void compute_features(const vector1d& antp, const int t1, vector1d& vals);
    void compute_features(const vector1dc& anlmtp, const int t1, vector1d& vals);

    void compute_prod_antp_deriv(const vector1d& antp, const int t1, vector1d& deriv);
    void compute_prod_anlmtp_deriv(
        const vector1dc& anlmtp, const int t1, vector1dc& deriv
    );

    void compute_prod_features(const vector1d& features, const int t1, vector1d& vals);

};

#endif
