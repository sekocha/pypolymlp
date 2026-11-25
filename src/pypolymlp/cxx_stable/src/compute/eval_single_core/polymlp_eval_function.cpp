/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_eval.h"


void PolymlpEval::_compute_anlmtp_singlecore(
    const vector1i& types,
    NeighborHalf& neigh,
    vector2dc& anlmtp)
{
    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    anlmtp = vector2dc(n_atom);

    vector2d anlmtp_r(n_atom), anlmtp_i(n_atom);
    for (int i = 0; i < n_atom; ++i) {
        int type1 = types[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;
        anlmtp_r[i] = vector1d(nlmtp_attrs_noconj.size(), 0.0);
        anlmtp_i[i] = vector1d(nlmtp_attrs_noconj.size(), 0.0);
    }

    int type1, type2, tp;
    double dx, dy, dz, dis;
    vector1d fn; vector1dc ylm; dc val;
    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const auto& nlmtp_attrs1 = nlmtp_attrs[type1];

        auto [begin, end] = neigh.range(i);
        for (int k = begin; k < end; ++k) {
            int j = neigh.neighbor_atom(k);
            neigh.diff_ij(k, dx, dy, dz);
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis >= fp.cutoff)
                continue;

            type2 = types[j];
            tp = type_pairs[type1][type2];
            const auto& params = tp_to_params[tp];
            get_fn_(dis, fp, params, fn);
            get_ylm_(dx, dy, dz, fp.maxl, ylm);

            const auto& attrs = nlmtp_attrs1[tp];
            for (const auto& nlmtp : attrs){
                if (fn[nlmtp.n_id] < 1e-20)
                    continue;
                const auto& lm_attr = nlmtp.lm;
                const int idx_i = nlmtp.ilocal_noconj_id;
                const int idx_j = nlmtp.jlocal_noconj_id;
                val = fn[nlmtp.n_id] * ylm[lm_attr.ylmkey];
                anlmtp_r[i][idx_i] += val.real();
                anlmtp_r[j][idx_j] += val.real() * lm_attr.sign_j;
                anlmtp_i[i][idx_i] += val.imag();
                anlmtp_i[j][idx_j] += val.imag() * lm_attr.sign_j;
            }
        }
    }
    for (int i = 0; i < n_atom; ++i) {
        polymlp_api.compute_anlmtp_conjugate(
            anlmtp_r[i], anlmtp_i[i], types[i], anlmtp[i]
        );
    }
}
