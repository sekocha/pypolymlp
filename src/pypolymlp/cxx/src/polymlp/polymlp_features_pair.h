/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES_PAIR
#define __POLYMLP_FEATURES_PAIR

//#include <map>

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_mapping.h"
#include "polymlp_model_params.h"

//#include "polymlp_features_polynomial.h"
//#include "polymlp_features_mapping_utils.h"
//#include "polymlp_features_utils.h"
//#include "polymlp_products.h"
//
//

class FeaturesPair {

    int n_type, n_variables;
    vector1i feature_sizes;

    Mapping mapping;
    ModelParams modelp;

    //FeaturesPairPoly poly;

    //vector3i prod, prod_deriv;
    //std::vector<MapFromVec> prod_map_deriv_from_keys;
    //std::vector<MappedMultipleFeaturesPair> mapped_features;
    //std::vector<MappedMultipleFeaturesPairDeriv> mapped_features_deriv;

    int set_linear_features_pair();
    int set_polynomial();

    //int set_mappings_standard();
    //int set_mappings_efficient(const feature_params& fp);
    //int set_deriv_mappings();
    //int set_mapped_features_deriv();

    public:

    FeaturesPair();
    FeaturesPair(const feature_params& fp);
    ~FeaturesPair();

    Maps& get_maps();
//    MapFromVec& get_prod_map_deriv(const int type1);
//    MapFromVec& get_prod_features_map(const int type1);
    const int get_n_variables() const;
//
//    void release_memory();
//
    void compute_features(const vector1d& antp, const int t1, vector1d& vals);
    void compute_prod_antp_deriv(const vector1d& antp, const int t1, vector1d& deriv);

//    void compute_prod_features(const vector1d& features, const int t1, vector1d& vals);

};

#endif
