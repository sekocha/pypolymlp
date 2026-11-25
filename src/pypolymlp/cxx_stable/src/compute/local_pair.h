/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __LOCAL_PAIR
#define __LOCAL_PAIR

#include <cmath>

#include "mlpcpp.h"
#include "compute/neighbor_full.h"
#include "polymlp/polymlp_api.h"
#include "polymlp/polymlp_functions_interface.h"


class LocalPair{

    int n_atom;

    public:

    LocalPair(const int n_atom);
    ~LocalPair();

    void pair(
        PolymlpAPI& polymlp,
        NeighborFull& neigh,
        const vector1i& types,
        const int atom1,
        vector1d& dn
    );

    void pair_d(
        PolymlpAPI& polymlp,
        NeighborFull& neigh,
        const vector1i& types,
        const int atom1,
        vector1d& dn,
        vector2d& dn_dfx,
        vector2d& dn_dfy,
        vector2d& dn_dfz,
        vector2d& dn_ds
    );

};

#endif
