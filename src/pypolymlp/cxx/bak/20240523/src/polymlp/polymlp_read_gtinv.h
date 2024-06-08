/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_READ_GTINV
#define __POLYMLP_READ_GTINV

#include "polymlp_gtinv_data.h"
#include "polymlp_gtinv_data_ver2.h"
#include "polymlp_mlpcpp.h"

class Readgtinv {

    vector2i l_array;
    vector3i lm_array;
    vector2d coeffs;

    void screening(const int& gtinv_order,
                   const vector1i& gtinv_maxl,
                   const std::vector<bool>& gtinv_sym,
                   const int& n_type);

    void screening_ver2(const int& gtinv_order,
                        const vector1i& gtinv_maxl,
                        const int& n_type);

    public:

    Readgtinv();
    Readgtinv(const int& gtinv_order,
              const vector1i& gtinv_maxl,
              const std::vector<bool>& gtinv_sym,
              const int& n_type,
              const int& version);
   ~Readgtinv();

    const vector3i& get_lm_seq() const;
    const vector2i& get_l_comb() const;
    const vector2d& get_lm_coeffs() const;

};

#endif
