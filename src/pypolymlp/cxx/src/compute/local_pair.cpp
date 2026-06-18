/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/local_pair.h"


LocalPair::LocalPair(const int n_atom_i){
    n_atom = n_atom_i;
}

LocalPair::~LocalPair(){}

void LocalPair::pair(
    PolymlpAPI& polymlp,
    NeighborFull& neigh,
    const vector1i& types,
    const int atom1,
    vector1d& dn
){

    const int type1 = types[atom1];
    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const auto& maps_type = maps.maps_type[type1];
    const auto& ntp_attrs = maps_type.ntp_attrs;

    dn = vector1d(ntp_attrs.size(), 0.0);
    double dx, dy, dz;
    vector1d fn;

    auto [begin, end] = neigh.range(atom1);
    for (int k = begin; k < end; ++k) {
        const int atom2 = neigh.neighbor_atom(k);
        // (dx, dy, dz) = pos[j] - pos[i]
        neigh.diff_ji(k, dx, dy, dz);
        double dis = sqrt(dx*dx + dy*dy + dz*dz);
        if (dis >= fp.cutoff)
            continue;

        const int type2 = types[atom2];
        const int tp = type_pairs[type1][type2];
        const auto& params = tp_to_params[tp];
        get_fn_(dis, fp, params, fn);
        for (const auto& ntp: ntp_attrs){
            if (tp == ntp.tp)
                dn[ntp.ilocal_id] += fn[ntp.n_id];
        }
    }
}

void LocalPair::pair_d(
    PolymlpAPI& polymlp,
    NeighborFull& neigh,
    const vector1i& types,
    const int atom1,
    vector1d& dn,
    vector2d& dn_dfx,
    vector2d& dn_dfy,
    vector2d& dn_dfz,
    vector2d& dn_ds
){

    const int type1 = types[atom1];
    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const auto& maps_type = maps.maps_type[type1];
    const auto& ntp_attrs = maps_type.ntp_attrs;

    dn = vector1d(ntp_attrs.size(), 0.0);
    dn_dfx = dn_dfy = dn_dfz = vector2d(ntp_attrs.size(), vector1d(n_atom, 0.0));
    dn_ds = vector2d(ntp_attrs.size(), vector1d(6, 0.0));

    int atom2, tp;
    double dis,dx,dy,dz,valx,valy,valz;
    vector1d fn,fn_d;

    auto [begin, end] = neigh.range(atom1);
    for (int k = begin; k < end; ++k) {
        const int atom2 = neigh.neighbor_atom(k);
        // (dx, dy, dz) = pos[j] - pos[i]
        neigh.diff_ji(k, dx, dy, dz);
        dis = sqrt(dx*dx + dy*dy + dz*dz);
        if (dis >= fp.cutoff)
            continue;

        const int type2 = types[atom2];
        const int tp = type_pairs[type1][type2];
        const auto& params = tp_to_params[tp];
        get_fn_(dis, fp, params, fn, fn_d);
        for (const auto& ntp: ntp_attrs){
            if (tp != ntp.tp)
                continue;

            const int idx_i = ntp.ilocal_id;
            dn[idx_i] += fn[ntp.n_id];
            valx = fn_d[ntp.n_id] * dx / dis;
            valy = fn_d[ntp.n_id] * dy / dis;
            valz = fn_d[ntp.n_id] * dz / dis;
            dn_dfx[idx_i][atom1] += valx;
            dn_dfy[idx_i][atom1] += valy;
            dn_dfz[idx_i][atom1] += valz;
            dn_dfx[idx_i][atom2] -= valx;
            dn_dfy[idx_i][atom2] -= valy;
            dn_dfz[idx_i][atom2] -= valz;
            dn_ds[idx_i][0] -= valx * dx;
            dn_ds[idx_i][1] -= valy * dy;
            dn_ds[idx_i][2] -= valz * dz;
            dn_ds[idx_i][3] -= valx * dy;
            dn_ds[idx_i][4] -= valy * dz;
            dn_ds[idx_i][5] -= valz * dx;
        }
    }
}
