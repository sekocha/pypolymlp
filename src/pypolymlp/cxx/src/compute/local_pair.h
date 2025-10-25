/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __LOCAL_PAIR
#define __LOCAL_PAIR

#include <cmath>

#include "mlpcpp.h"
#include "polymlp/polymlp_api.h"
#include "polymlp/polymlp_functions_interface.h"


class LocalPair{

    int n_atom;

    public:

    LocalPair(const int n_atom);
    ~LocalPair();

    void pair(
        PolymlpAPI& polymlp,
        const int type1,
        const vector2d& dis_a,
        vector1d& dn
    );

    void pair_d(
        PolymlpAPI& polymlp,
        const int atom1,
        const int type1,
        const vector2d& dis_a,
        const vector3d& diff_a,
        const vector2i& atom2_a,
        vector1d& dn,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
    );

};

#endif
