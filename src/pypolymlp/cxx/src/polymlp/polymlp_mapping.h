/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_MAPPING
#define __POLYMLP_MAPPING

#include "polymlp_mlpcpp.h"
#include "polymlp_structs.h"


class Mapping {

    int n_type, n_fn, maxl;
    Maps maps;

    void set_type_pairs(const feature_params& fp);
    void set_type_pairs_charge(const feature_params& fp);
    void set_map_n_to_tplist();

    void set_ntp_global_attrs();
    void set_ntp_local_attrs();

    void set_lm_attrs();
    void set_nlmtp_global_attrs();
    void set_nlmtp_local_attrs();
    void set_nlmtp_local_conj_ids();

    public:

    Mapping();
    Mapping(const struct feature_params& fp);
    ~Mapping();

    Maps& get_maps();

};

#endif
