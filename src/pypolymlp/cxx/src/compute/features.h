/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __FEATURE
#define __FEATURE

#include "mlpcpp.h"
#include "polymlp/polymlp_mapping.h"
#include "polymlp/polymlp_model_params.h"
#include "polymlp/polymlp_features.h"


struct FeatureSingleTerm {
    double coeff;
    int prod_key;
};

struct FeatureSingleTermDeriv {
    double coeff;
    int prod_key;
    int nlmtp_key;
};

struct PolynomialTerm {
    int seq_id;
    vector1i comb_tlocal;
};


typedef std::vector<std::vector<FeatureSingleTerm> > FeatureVector;
typedef std::vector<std::vector<FeatureSingleTermDeriv> > FeatureVectorDeriv;
typedef std::vector<PolynomialTerm> Polynomial;
typedef std::unordered_map<vector1i,int,HashVI> ProdMapFromKeys;


class FunctionFeatures {

    Mapping mapping;
    ModelParams modelp;
    int n_type;

    vector3i prod_map, prod_map_deriv;
    std::vector<ProdMapFromKeys> prod_map_from_keys, prod_map_deriv_from_keys;
    std::vector<FeatureVector> linear_features;
    std::vector<FeatureVectorDeriv> linear_features_deriv;

    std::vector<Polynomial> polynomials1, polynomials2, polynomials3;

    vector1i erase_a_key(const vector1i& original, const int idx);
    void nonequiv_set_to_mappings(
        const std::set<vector1i>& nonequiv_keys,
        ProdMapFromKeys& map_from_keys,
        vector2i& map
    );

    void set_mapping_prod(const Features& f_obj);
    void set_features_using_mappings(const Features& f_obj);

    void set_polynomials(const ModelParams& modelp);

    // deprecated
    void set_features_using_mappings_simple(const Features& f_obj);
    // not needed ?
    void sort_linear_features_deriv();

    public:

    FunctionFeatures();
    FunctionFeatures(const Features& f_obj);
    ~FunctionFeatures();

    const Mapping& get_mapping() const;
    const ModelParams& get_model_params() const;
    const vector2i& get_prod_map(const int t) const;
    const vector2i& get_prod_map_deriv(const int t) const;
    const FeatureVector& get_linear_features(const int t) const;
    const FeatureVectorDeriv& get_linear_features_deriv(const int t) const;

    const Polynomial& get_polynomial1(const int t) const;
    const Polynomial& get_polynomial2(const int t) const;
    const Polynomial& get_polynomial3(const int t) const;
};

#endif
