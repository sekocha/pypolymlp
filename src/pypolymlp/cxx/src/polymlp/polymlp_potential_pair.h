/****************************************************************************

        Copyright (C) 2026 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_POTENTIAL_PAIR
#define __POLYMLP_POTENTIAL_PAIR

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"
#include "polymlp_features_pair.h"


struct ForceTerm1 {
    double coeff;
};
struct ForceTerm2 {
    double coeff;
    int local_id1;
};
struct ForceTerm3 {
    double coeff;
    int local_id1;
    int local_id2;
};

class PotentialPair {

    int n_type;
    FeaturesPair f_obj;

    std::vector<std::vector<ForceTerm1>> force_terms1;
    std::vector<std::vector<ForceTerm2>> force_terms2;
    std::vector<std::vector<ForceTerm3>> force_terms3;

    int set_force_terms(const vector1d& pot);

    public:

    PotentialPair();
    PotentialPair(const feature_params& fp, const vector1d& pot);
    ~PotentialPair();

    Maps& get_maps();
    void compute_sum(
        const vector1d& antp,
        const int type1,
        vector1d& prod_sum_e,
        vector1d& prod_sum_f
    );

//    int convert_unit(const double energy_conv);

};

#endif
