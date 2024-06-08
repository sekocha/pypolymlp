/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __FEATURE
#define __FEATURE

#include "mlpcpp.h"
#include "polymlp/polymlp_features.h"
#include "polymlp/polymlp_potential.h"


struct FeatureSingleTerm {
    double coeff;
    int prod_key;
};

struct FeatureSingleTermDeriv {
    double coeff;
    int prod_key;
    int nlmtc_key;
};

typedef std::vector<std::vector<FeatureSingleTerm> > FeatureVector;
typedef std::vector<std::vector<FeatureSingleTermDeriv> > FeatureVectorDeriv;
typedef std::unordered_map<vector1i,int,HashVI> ProdMapFromKeys;


class FunctionFeatures {

    std::vector<lmAttribute> lm_map;
    std::vector<nlmtcAttribute> nlmtc_map_no_conjugate, nlmtc_map;
    std::vector<ntcAttribute> ntc_map;

    bool eliminate_conj;
    int n_nlmtc_all, n_type;

    vector3i prod_map, prod_map_deriv;
    std::vector<ProdMapFromKeys> prod_map_from_keys, prod_map_deriv_from_keys;
    std::vector<FeatureVector> linear_features;
    std::vector<FeatureVectorDeriv> linear_features_deriv;

    vector1i erase_a_key(const vector1i& original, const int idx);
    void nonequiv_set_to_mappings(const std::set<vector1i>& nonequiv_keys,
                                  ProdMapFromKeys& map_from_keys,
                                  vector2i& map);

    void set_mapping_prod(const Features& f_obj);
    void set_features_using_mappings(const Features& f_obj);

    // deprecated
    void set_features_using_mappings_simple(const Features& f_obj);
    // not needed ?
    void sort_linear_features_deriv();

    public:

    FunctionFeatures();
    FunctionFeatures(const Features& f_obj);
    ~FunctionFeatures();

    const std::vector<lmAttribute>& get_lm_map() const;
    const std::vector<nlmtcAttribute>& get_nlmtc_map_no_conjugate() const;
    const std::vector<nlmtcAttribute>& get_nlmtc_map() const;
    const std::vector<ntcAttribute>& get_ntc_map() const;
    const int get_n_nlmtc_all() const;

    const vector2i& get_prod_map(const int t) const;
    const vector2i& get_prod_map_deriv(const int t) const;
    const FeatureVector& get_linear_features(const int t) const;
    const FeatureVectorDeriv& get_linear_features_deriv(const int t) const;

};

#endif
