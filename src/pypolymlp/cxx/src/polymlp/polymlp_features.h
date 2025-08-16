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
#include "polymlp_features_mapping_utils.h"
#include "polymlp_features_utils.h"
#include "polymlp_products.h"


class Features {

    int n_type;
    bool eliminate_conj, set_features_deriv;
    vector1i feature_sizes;

    Mapping mapping;
    ModelParams modelp;
    FeaturesPoly poly;

    vector3i prod, prod_deriv;
    std::vector<MapFromVec> prod_map_deriv_from_keys;
    std::vector<MappedMultipleFeatures> mapped_features;
    std::vector<MappedMultipleFeaturesDeriv> mapped_features_deriv;

    int set_mappings_standard();
    int set_mappings_efficient(const feature_params& fp);
    int set_deriv_mappings();
    int set_mapped_features_deriv();

    void compute_features_deriv_single_component(
        const vector2dc& anlmtp_d,
        const int type1,
        const vector1dc& prod_anlmtp_d,
        vector2d& dn_d
    );

    public:

    Features();
    Features(const feature_params& fp, const bool set_deriv);
    ~Features();

    Maps& get_maps();
    MapFromVec& get_prod_map_deriv(const int type1);
    MapFromVec& get_prod_features_map(const int type1);
    const int get_n_variables() const;

    void release_memory();

    void compute_features(const vector1d& antp, const int t1, vector1d& vals);
    void compute_features(const vector1dc& anlmtp, const int t1, vector1d& vals);

    void compute_features_deriv(
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

    void compute_prod_antp_deriv(const vector1d& antp, const int t1, vector1d& deriv);
    void compute_prod_anlmtp_deriv(
        const vector1dc& anlmtp, const int t1, vector1dc& deriv
    );

    void compute_prod_features(const vector1d& features, const int t1, vector1d& vals);

};

#endif
