/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/local.h"


const double tol = 1e-20;

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
                    if (tp == nlmtp.tp and fabs(fn[nlmtp.n_id]) > tol){
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
    const auto& nlmtp_attrs = maps_type.nlmtp_attrs;
    const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

    anlmtp = vector1dc(nlmtp_attrs.size(), 0.0);
    anlmtp_dfx = vector2dc(nlmtp_attrs.size(), vector1dc(n_atom, 0.0));
    anlmtp_dfy = vector2dc(nlmtp_attrs.size(), vector1dc(n_atom, 0.0));
    anlmtp_dfz = vector2dc(nlmtp_attrs.size(), vector1dc(n_atom, 0.0));
    anlmtp_ds = vector2dc(nlmtp_attrs.size(), vector1dc(6, 0.0));

    int atom2;
    double delx,dely,delz,dis,cc;
    dc d1,val,valx,valy,valz;
    vector1d fn, fn_d;
    vector1dc ylm,ylm_dx,ylm_dy,ylm_dz;

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
                    if (tp == nlmtp.tp and fabs(fn[nlmtp.n_id]) > tol){
                        const auto& lm_attr = nlmtp.lm;
                        const int ylmkey = lm_attr.ylmkey;
                        const int idx_i = nlmtp.ilocal_id;
                        val = fn[nlmtp.n_id] * ylm[ylmkey];
                        anlmtp[idx_i] += val;

                        d1 = fn_d[nlmtp.n_id] * ylm[ylmkey] / dis;
                        valx = (d1 * delx + fn[nlmtp.n_id] * ylm_dx[ylmkey]);
                        valy = (d1 * dely + fn[nlmtp.n_id] * ylm_dy[ylmkey]);
                        valz = (d1 * delz + fn[nlmtp.n_id] * ylm_dz[ylmkey]);

                        anlmtp_dfx[idx_i][atom1] += valx;
                        anlmtp_dfx[idx_i][atom2] -= valx;
                        anlmtp_dfy[idx_i][atom1] += valy;
                        anlmtp_dfy[idx_i][atom2] -= valy;
                        anlmtp_dfz[idx_i][atom1] += valz;
                        anlmtp_dfz[idx_i][atom2] -= valz;

                        anlmtp_ds[idx_i][0] -= valx * delx;
                        anlmtp_ds[idx_i][1] -= valy * dely;
                        anlmtp_ds[idx_i][2] -= valz * delz;
                        anlmtp_ds[idx_i][3] -= valx * dely;
                        anlmtp_ds[idx_i][4] -= valy * delz;
                        anlmtp_ds[idx_i][5] -= valz * delx;
                    }
                }
            }
        }
    }

    for (const auto& nlmtp: nlmtp_attrs){
        if (nlmtp.lm.conj == true){
            const auto cc_coeff = nlmtp.lm.cc_coeff;
            const int id1 = nlmtp.ilocal_id;
            const int id2 = nlmtp.ilocal_conj_id;
            anlmtp[id1] = cc_coeff * std::conj(anlmtp[id2]);
            for (size_t k = 0; k < n_atom; ++k)
                anlmtp_dfx[id1][k] = cc_coeff * std::conj(anlmtp_dfx[id2][k]);
            for (size_t k = 0; k < n_atom; ++k)
                anlmtp_dfy[id1][k] = cc_coeff * std::conj(anlmtp_dfy[id2][k]);
            for (size_t k = 0; k < n_atom; ++k)
                anlmtp_dfz[id1][k] = cc_coeff * std::conj(anlmtp_dfz[id2][k]);
            for (size_t k = 0; k < 6; ++k)
                anlmtp_ds[id1][k] = cc_coeff * std::conj(anlmtp_ds[id2][k]);
        }
    }
}
