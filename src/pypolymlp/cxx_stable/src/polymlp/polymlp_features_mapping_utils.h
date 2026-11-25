/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_FEATURES_MAPPING_UTILS
#define __POLYMLP_FEATURES_MAPPING_UTILS


#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"


int get_nonequiv_ids(const MultipleFeatures& features, std::set<vector1i>& nonequiv);

int get_nonequiv_deriv_ids(
    const MultipleFeatures& features,
    MapsType& maps_type,
    const bool eliminate_conj,
    std::set<vector1i>& nonequiv
);

int _convert_single_feature_to_map(
    const SingleFeature& sfeature,
    MapFromVec& prod_map_from_keys,
    std::unordered_map<int, double>& sfeature_map
);

int convert_to_mapped_features_algo1(
    const MultipleFeatures& features,
    MapFromVec& prod_map_from_keys,
    const vector2i& prod,
    MappedMultipleFeatures& mapped_features
);

int convert_to_mapped_features_algo2(
    const MultipleFeatures& features,
    MapFromVec& prod_map_from_keys,
    MappedMultipleFeatures& mapped_features
);

int convert_to_mapped_features_deriv_algo1(
    const MultipleFeatures& features,
    MapFromVec& prod_map_deriv_from_keys,
    const vector2i& prod_deriv,
    MappedMultipleFeaturesDeriv& mapped_features_deriv
);

int convert_to_mapped_features_deriv_algo2(
    const MultipleFeatures& features,
    MapFromVec& prod_map_deriv_from_keys,
    MappedMultipleFeaturesDeriv& mapped_features_deriv
);

void sort(MappedMultipleFeaturesDeriv& mapped_features_deriv);

#endif
