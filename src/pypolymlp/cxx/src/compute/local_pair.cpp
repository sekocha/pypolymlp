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
    const int type1,
    const vector2d& dis_a,
    vector1d& dn
){

    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_type = fp.n_type;
    const auto& maps_type = maps.maps_type[type1];
    const auto& ntp_attrs = maps_type.ntp_attrs;

    dn = vector1d(ntp_attrs.size(), 0.0);
    int tp;
    vector1d fn;
    for (int type2 = 0; type2 < n_type; ++type2){
        tp = type_pairs[type1][type2];
        for (const auto& dis: dis_a[type2]){
            const auto& params = tp_to_params[tp];
            get_fn_(dis, fp, params, fn);
            for (const auto& ntp: ntp_attrs){
                if (tp == ntp.tp) dn[ntp.ilocal_id] += fn[ntp.n_id];
            }
        }
    }
}

void LocalPair::pair_d(
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
){

    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_type = fp.n_type;
    const auto& maps_type = maps.maps_type[type1];
    const auto& ntp_attrs = maps_type.ntp_attrs;

    dn = vector1d(ntp_attrs.size(), 0.0);
    dn_dfx = dn_dfy = dn_dfz = vector2d(ntp_attrs.size(), vector1d(n_atom, 0.0));
    dn_ds = vector2d(ntp_attrs.size(), vector1d(6, 0.0));

    int atom2, tp;
    double dis,delx,dely,delz,valx,valy,valz;
    vector1d fn,fn_d;
    for (int type2 = 0; type2 < n_type; ++type2){
        tp = type_pairs[type1][type2];
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            delx = diff_a[type2][j][0];
            dely = diff_a[type2][j][1];
            delz = diff_a[type2][j][2];
            atom2 = atom2_a[type2][j];
            const auto& params = tp_to_params[tp];
            get_fn_(dis, fp, params, fn, fn_d);
            for (const auto& ntp: ntp_attrs){
                if (tp == ntp.tp){
                    const int idx_i = ntp.ilocal_id;
                    dn[idx_i] += fn[ntp.n_id];
                    valx = fn_d[ntp.n_id] * delx / dis;
                    valy = fn_d[ntp.n_id] * dely / dis;
                    valz = fn_d[ntp.n_id] * delz / dis;
                    dn_dfx[idx_i][atom1] += valx;
                    dn_dfy[idx_i][atom1] += valy;
                    dn_dfz[idx_i][atom1] += valz;
                    dn_dfx[idx_i][atom2] -= valx;
                    dn_dfy[idx_i][atom2] -= valy;
                    dn_dfz[idx_i][atom2] -= valz;
                    dn_ds[idx_i][0] -= valx * delx;
                    dn_ds[idx_i][1] -= valy * dely;
                    dn_ds[idx_i][2] -= valz * delz;
                    dn_ds[idx_i][3] -= valx * dely;
                    dn_ds[idx_i][4] -= valy * delz;
                    dn_ds[idx_i][5] -= valz * delx;
                }
            }
        }
    }
}
