/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/local.h"


Local::Local(){}
Local::Local(const int n_atom_i) : n_atom(n_atom_i) {}

Local::~Local(){}

void Local::gtinv(
    PolymlpAPI& polymlp,
    const int type1,
    const vector2d& dis_a,
    const vector3d& diff_a,
    vector1d& dn
){

    vector1dc anlmtp;
    compute_anlmtp(polymlp, type1, dis_a, diff_a, anlmtp);
    polymlp.compute_features(anlmtp, type1, dn);
}


void Local::gtinv_d(
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

    vector1dc anlmtp;
    vector2dc anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds;
    compute_anlmtp_d(
        polymlp, atom1, type1, dis_a, diff_a, atom2_a,
        anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds
    );

    polymlp.compute_features(anlmtp, type1, dn);
    polymlp.compute_features_deriv(
        anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds,
        type1, dn_dfx, dn_dfy, dn_dfz, dn_ds
    );
}


void Local::compute_anlmtp(
    PolymlpAPI& polymlp,
    const int type1,
    const vector2d& dis_a,
    const vector3d& diff_a,
    vector1dc& anlmtp
){

    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_type = fp.n_type;
    const auto& maps_type = maps.maps_type[type1];
    const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

    vector1d anlmtp_r(nlmtp_attrs_noconj.size(), 0.0);
    vector1d anlmtp_i(nlmtp_attrs_noconj.size(), 0.0);

    int ylmkey;
    double dis;
    dc val;
    vector1d fn;
    vector1dc ylm;

    for (int type2 = 0; type2 < n_type; ++type2){
        const int tp = type_pairs[type1][type2];
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            if (dis < fp.cutoff){
                const vector1d &sph = cartesian_to_spherical_(diff_a[type2][j]);
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn);
                get_ylm_(sph[0], sph[1], fp.maxl, ylm);
                for (const auto& nlmtp: nlmtp_attrs_noconj){
                    if (tp == nlmtp.tp){
                        const auto& lm_attr = nlmtp.lm;
                        const int idx_i = nlmtp.ilocal_noconj_id;
                        val = fn[nlmtp.n_id] * ylm[lm_attr.ylmkey];
                        anlmtp_r[idx_i] += val.real();
                        anlmtp_i[idx_i] += val.imag();
                    }
                }
            }
        }
    }
    polymlp.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, type1, anlmtp);
}


void Local::compute_anlmtp_d(
    PolymlpAPI& polymlp,
    const int atom1,
    const int type1,
    const vector2d& dis_a,
    const vector3d& diff_a,
    const vector2i& atom2_a,
    vector1dc& anlmtp,
    vector2dc& anlmtp_dfx,
    vector2dc& anlmtp_dfy,
    vector2dc& anlmtp_dfz,
    vector2dc& anlmtp_ds
){
    const auto& fp = polymlp.get_fp();
    const auto& maps = polymlp.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_type = fp.n_type;
    const auto& maps_type = maps.maps_type[type1];
    const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

    vector1d anlmtp_r(nlmtp_attrs_noconj.size(), 0.0);
    vector1d anlmtp_i(nlmtp_attrs_noconj.size(), 0.0);
    vector2d anlmtp_dfx_r(nlmtp_attrs_noconj.size(), vector1d(n_atom, 0.0));
    vector2d anlmtp_dfx_i(nlmtp_attrs_noconj.size(), vector1d(n_atom, 0.0));
    vector2d anlmtp_dfy_r(nlmtp_attrs_noconj.size(), vector1d(n_atom, 0.0));
    vector2d anlmtp_dfy_i(nlmtp_attrs_noconj.size(), vector1d(n_atom, 0.0));
    vector2d anlmtp_dfz_r(nlmtp_attrs_noconj.size(), vector1d(n_atom, 0.0));
    vector2d anlmtp_dfz_i(nlmtp_attrs_noconj.size(), vector1d(n_atom, 0.0));
    vector2d anlmtp_ds_r(nlmtp_attrs_noconj.size(), vector1d(6, 0.0));
    vector2d anlmtp_ds_i(nlmtp_attrs_noconj.size(), vector1d(6, 0.0));

    vector1d fn, fn_d;
    vector1dc ylm,ylm_dx,ylm_dy,ylm_dz;
    double delx,dely,delz,dis,cc;
    dc d1,val,valx,valy,valz,val0,val1,val2,val3,val4,val5;
    int atom2;

    for (int type2 = 0; type2 < n_type; ++type2){
        const int tp = type_pairs[type1][type2];
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            delx = diff_a[type2][j][0];
            dely = diff_a[type2][j][1];
            delz = diff_a[type2][j][2];
            if (dis < fp.cutoff){
                atom2 = atom2_a[type2][j];
                const vector1d &sph = cartesian_to_spherical_(diff_a[type2][j]);
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], fp.maxl, ylm, ylm_dx, ylm_dy, ylm_dz);
                for (const auto& nlmtp: nlmtp_attrs_noconj){
                    if (tp == nlmtp.tp){
                        const auto& lm_attr = nlmtp.lm;
                        const int ylmkey = lm_attr.ylmkey;
                        const int idx_i = nlmtp.ilocal_noconj_id;
                        val = fn[nlmtp.n_id] * ylm[ylmkey];
                        anlmtp_r[idx_i] += val.real();
                        anlmtp_i[idx_i] += val.imag();

                        d1 = fn_d[nlmtp.n_id] * ylm[ylmkey] / dis;
                        valx = (d1 * delx + fn[nlmtp.n_id] * ylm_dx[ylmkey]);
                        valy = (d1 * dely + fn[nlmtp.n_id] * ylm_dy[ylmkey]);
                        valz = (d1 * delz + fn[nlmtp.n_id] * ylm_dz[ylmkey]);

                        anlmtp_dfx_r[idx_i][atom1] += valx.real();
                        anlmtp_dfy_r[idx_i][atom1] += valy.real();
                        anlmtp_dfz_r[idx_i][atom1] += valz.real();
                        anlmtp_dfx_i[idx_i][atom1] += valx.imag();
                        anlmtp_dfy_i[idx_i][atom1] += valy.imag();
                        anlmtp_dfz_i[idx_i][atom1] += valz.imag();

                        anlmtp_dfx_r[idx_i][atom2] -= valx.real();
                        anlmtp_dfy_r[idx_i][atom2] -= valy.real();
                        anlmtp_dfz_r[idx_i][atom2] -= valz.real();
                        anlmtp_dfx_i[idx_i][atom2] -= valx.imag();
                        anlmtp_dfy_i[idx_i][atom2] -= valy.imag();
                        anlmtp_dfz_i[idx_i][atom2] -= valz.imag();

                        val0 -= valx * delx;
                        val1 -= valy * dely;
                        val2 -= valz * delz;
                        val3 -= valx * dely;
                        val4 -= valy * delz;
                        val5 -= valz * delx;

                        anlmtp_ds_r[idx_i][0] -= val0.real();
                        anlmtp_ds_r[idx_i][1] -= val1.real();
                        anlmtp_ds_r[idx_i][2] -= val2.real();
                        anlmtp_ds_r[idx_i][3] -= val3.real();
                        anlmtp_ds_r[idx_i][4] -= val4.real();
                        anlmtp_ds_r[idx_i][5] -= val5.real();
                        anlmtp_ds_i[idx_i][0] -= val0.imag();
                        anlmtp_ds_i[idx_i][1] -= val1.imag();
                        anlmtp_ds_i[idx_i][2] -= val2.imag();
                        anlmtp_ds_i[idx_i][3] -= val3.imag();
                        anlmtp_ds_i[idx_i][4] -= val4.imag();
                        anlmtp_ds_i[idx_i][5] -= val5.imag();
                    }
                }
            }
        }
    }

    polymlp.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, type1, anlmtp);
    polymlp.compute_anlmtp_conjugate(anlmtp_dfx_r, anlmtp_dfx_i, type1, anlmtp_dfx);
    polymlp.compute_anlmtp_conjugate(anlmtp_dfy_r, anlmtp_dfy_i, type1, anlmtp_dfy);
    polymlp.compute_anlmtp_conjugate(anlmtp_dfz_r, anlmtp_dfz_i, type1, anlmtp_dfz);
    polymlp.compute_anlmtp_conjugate(anlmtp_ds_r, anlmtp_ds_i, type1, anlmtp_ds);

}

/*
void Local::compute_linear_features_deriv(
    const vector1dc& prod_anlmtp_d,
    const FunctionFeatures& features,
    const vector2dc& anlmtp_dfx,
    const vector2dc& anlmtp_dfy,
    const vector2dc& anlmtp_dfz,
    const vector2dc& anlmtp_ds,
    vector2d& dn_dfx,
    vector2d& dn_dfy,
    vector2d& dn_dfz,
    vector2d& dn_ds
){

    const auto& linear_features_d = features.get_linear_features_deriv(type1);
    int size(0);
    for (const auto& sfeature: linear_features_d){
        if (sfeature.size() > 0) ++size;
    }

    dn_dfx = vector2d(size, vector1d(n_atom, 0.0));
    dn_dfy = vector2d(size, vector1d(n_atom, 0.0));
    dn_dfz = vector2d(size, vector1d(n_atom, 0.0));
    dn_ds = vector2d(size, vector1d(6, 0.0));

    int nlmtp_key, idx(0);
    dc val_dc;
    for (const auto& sfeature: linear_features_d){
        if (sfeature.size() > 0){
            for (const auto& sterm: sfeature){
                val_dc = sterm.coeff * prod_anlmtp_d[sterm.prod_key];
                nlmtp_key = sterm.nlmtp_key;
                for (int j = 0; j < n_atom; ++j){
                    dn_dfx[idx][j] +=
                        prod_real(val_dc, anlmtp_dfx[nlmtp_key][j]);
                    dn_dfy[idx][j] +=
                        prod_real(val_dc, anlmtp_dfy[nlmtp_key][j]);
                    dn_dfz[idx][j] +=
                        prod_real(val_dc, anlmtp_dfz[nlmtp_key][j]);
                }
                for (int j = 0; j < 6; ++j){
                    dn_ds[idx][j] += prod_real(val_dc, anlmtp_ds[nlmtp_key][j]);
                }
            }
            ++idx;
        }
    }
}
*/
