/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/local_fast.h"

LocalFast::LocalFast(){}

LocalFast::LocalFast(const int& n_atom_i,
                     const int& atom1_i,
                     const int& type1_i,
                     const struct feature_params& fp_i,
                     const FunctionFeatures& features){

    n_atom = n_atom_i, atom1 = atom1_i, type1 = type1_i,
    fp = fp_i;
    n_type = fp.n_type;

    const auto& mapping = features.get_mapping();
    type_pairs = mapping.get_type_pairs()[type1];
}

LocalFast::~LocalFast(){}

void LocalFast::pair(const vector2d& dis_a,
                     const FunctionFeatures& features,
                     vector1d& dn){

    const auto& mapping = features.get_mapping();
    const auto& ntp_attrs = mapping.get_ntp_attrs(type1);
    const auto& tp_to_params = mapping.get_type_pair_to_params();

    dn = vector1d(ntp_attrs.size(), 0.0);
    vector2d params;
    for (int type2 = 0; type2 < n_type; ++type2){
        const int tp = type_pairs[type2];
        for (const auto& dis: dis_a[type2]){
            params = tp_to_params[tp];
            vector1d fn;
            get_fn_(dis, fp, params, fn);
            int key(0);
            for (const auto& ntp: ntp_attrs){
                if (tp == ntp.tp) dn[key] += fn[ntp.n_id];
                ++key;
            }
        }
    }
}

void LocalFast::pair_d(const vector2d& dis_a,
                       const vector3d& diff_a,
                       const vector2i& atom2_a,
                      const FunctionFeatures& features,
                       vector1d& dn,
                       vector2d& dn_dfx,
                       vector2d& dn_dfy,
                       vector2d& dn_dfz,
                       vector2d& dn_ds){

    const auto& mapping = features.get_mapping();
    const auto& ntp_attrs = mapping.get_ntp_attrs(type1);
    const auto& tp_to_params = mapping.get_type_pair_to_params();

    dn = vector1d(ntp_attrs.size(), 0.0);
    dn_dfx = dn_dfy = dn_dfz = vector2d(ntp_attrs.size(), vector1d(n_atom, 0.0));
    dn_ds = vector2d(ntp_attrs.size(), vector1d(6, 0.0));

    int atom2, col;
    double dis,delx,dely,delz,valx,valy,valz;
    vector2d params;

    for (int type2 = 0; type2 < n_type; ++type2){
        const int tp = type_pairs[type2];
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            delx = diff_a[type2][j][0];
            dely = diff_a[type2][j][1];
            delz = diff_a[type2][j][2];
            atom2 = atom2_a[type2][j];

            params = tp_to_params[tp];
            vector1d fn,fn_d;
            get_fn_(dis, fp, params, fn, fn_d);
            col = 0;
            for (const auto& ntp: ntp_attrs){
                if (tp == ntp.tp){
                    dn[col] += fn[ntp.n_id];
                    valx = fn_d[ntp.n_id] * delx / dis;
                    valy = fn_d[ntp.n_id] * dely / dis;
                    valz = fn_d[ntp.n_id] * delz / dis;
                    dn_dfx[col][atom1] += valx;
                    dn_dfy[col][atom1] += valy;
                    dn_dfz[col][atom1] += valz;
                    dn_dfx[col][atom2] -= valx;
                    dn_dfy[col][atom2] -= valy;
                    dn_dfz[col][atom2] -= valz;
                    dn_ds[col][0] -= valx * delx;
                    dn_ds[col][1] -= valy * dely;
                    dn_ds[col][2] -= valz * delz;
                    dn_ds[col][3] -= valx * dely;
                    dn_ds[col][4] -= valy * delz;
                    dn_ds[col][5] -= valz * delx;
                }
                ++col;
            }
        }
    }
}

void LocalFast::gtinv(const vector2d& dis_a,
                      const vector3d& diff_a,
                      const FunctionFeatures& features,
                      vector1d& dn){

    const auto& prod_map = features.get_prod_map(type1);

    vector1dc anlmtp;
    compute_anlm(dis_a, diff_a, features, anlmtp);

    vector1d prod_anlmtp;
    compute_products_real(prod_map, anlmtp, prod_anlmtp);

    compute_linear_features(prod_anlmtp, features, dn);
}


void LocalFast::gtinv_d(const vector2d& dis_a,
                        const vector3d& diff_a,
                        const vector2i& atom2_a,
                        const FunctionFeatures& features,
                        vector1d& dn,
                        vector2d& dn_dfx,
                        vector2d& dn_dfy,
                        vector2d& dn_dfz,
                        vector2d& dn_ds){

    const auto& prod_map = features.get_prod_map(type1);
    const auto& prod_map_d = features.get_prod_map_deriv(type1);

    vector1dc anlmtp;
    vector2dc anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds;
    compute_anlm_d(
        dis_a, diff_a, atom2_a, features,
        anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds
    );

    vector1d prod_anlmtp;
    vector1dc prod_anlmtp_d;
    compute_products_real(prod_map, anlmtp, prod_anlmtp);
    compute_products<dc>(prod_map_d, anlmtp, prod_anlmtp_d);

    compute_linear_features(prod_anlmtp, features, dn);
    compute_linear_features_deriv(
        prod_anlmtp_d, features,
        anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds,
        dn_dfx, dn_dfy, dn_dfz, dn_ds
    );

}

void LocalFast::compute_linear_features(const vector1d& prod_anlmtp,
                                        const FunctionFeatures& features,
                                        vector1d& dn){

    const auto& linear_features = features.get_linear_features(type1);
    int size(0);
    for (const auto& sfeature: linear_features){
        if (sfeature.size() > 0) ++size;
    }
    dn = vector1d(size, 0.0);

    int idx = 0;
    double val;
    for (const auto& sfeature: linear_features){
        if (sfeature.size() > 0){
            val = 0.0;
            for (const auto& sterm: sfeature){
                val += sterm.coeff * prod_anlmtp[sterm.prod_key];
            }
            dn[idx] = val;
            ++idx;
        }
    }
}

void LocalFast::compute_linear_features_deriv(
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


void LocalFast::compute_anlm(const vector2d& dis_a,
                             const vector3d& diff_a,
                             const FunctionFeatures& features,
                             vector1dc& anlm){

    const auto& mapping = features.get_mapping();
    const auto& nlmtp_attrs_no_conj = mapping.get_nlmtp_attrs_no_conjugate();
    const int n_nlmtp_all = mapping.get_n_nlmtp_all();
    const auto& tp_to_params = mapping.get_type_pair_to_params();
    anlm = vector1dc(n_nlmtp_all, 0.0);

    vector1dc ylm;
    int ylmkey;
    double dis,cc;
    vector2d params;

    for (int type2 = 0; type2 < n_type; ++type2){
        const int tp = type_pairs[type2];
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            if (dis < fp.cutoff){
                const vector1d &sph = cartesian_to_spherical_(diff_a[type2][j]);
                params = tp_to_params[tp];
                vector1d fn;
                get_fn_(dis, fp, params, fn);
                get_ylm_(sph[0], sph[1], fp.maxl, ylm);
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    if (tp == nlmtp.tp){
                        ylmkey = nlmtp.lm.ylmkey;
                        anlm[nlmtp.nlmtp_key] += fn[nlmtp.n_id] * ylm[ylmkey];
                    }
                }
            }
        }
    }

    for (const auto& nlmtp: nlmtp_attrs_no_conj){
        cc = nlmtp.lm.cc_coeff;
        anlm[nlmtp.conj_key] = cc * std::conj(anlm[nlmtp.nlmtp_key]);
    }
}

void LocalFast::compute_anlm_d(const vector2d& dis_a,
                               const vector3d& diff_a,
                               const vector2i& atom2_a,
                               const FunctionFeatures& features,
                               vector1dc& anlm,
                               vector2dc& anlm_dfx,
                               vector2dc& anlm_dfy,
                               vector2dc& anlm_dfz,
                               vector2dc& anlm_ds){

    const auto& mapping = features.get_mapping();
    const auto& nlmtp_attrs_no_conj = mapping.get_nlmtp_attrs_no_conjugate();
    const int n_nlmtp_all = mapping.get_n_nlmtp_all();
    const auto& tp_to_params = mapping.get_type_pair_to_params();

    anlm = vector1dc(n_nlmtp_all, 0.0);
    anlm_dfx = vector2dc(n_nlmtp_all, vector1dc(n_atom, 0.0));
    anlm_dfy = vector2dc(n_nlmtp_all, vector1dc(n_atom, 0.0));
    anlm_dfz = vector2dc(n_nlmtp_all, vector1dc(n_atom, 0.0));
    anlm_ds = vector2dc(n_nlmtp_all, vector1dc(6, 0.0));

    vector1dc ylm,ylm_dx,ylm_dy,ylm_dz;
    double delx,dely,delz,dis,cc;
    dc d1,valx,valy,valz;
    int atom2, ylmkey;
    vector2d params;

    for (int type2 = 0; type2 < n_type; ++type2){
        const int tp = type_pairs[type2];
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            delx = diff_a[type2][j][0];
            dely = diff_a[type2][j][1];
            delz = diff_a[type2][j][2];
            if (dis < fp.cutoff){
                atom2 = atom2_a[type2][j];
                const vector1d &sph = cartesian_to_spherical_(diff_a[type2][j]);
                params = tp_to_params[tp];
                vector1d fn, fn_d;
                get_fn_(dis, fp, params, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], fp.maxl,
                         ylm, ylm_dx, ylm_dy, ylm_dz);
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    if (tp == nlmtp.tp){
                        ylmkey = nlmtp.lm.ylmkey;
                        anlm[nlmtp.nlmtp_key] += fn[nlmtp.n_id] * ylm[ylmkey];
                        d1 = fn_d[nlmtp.n_id] * ylm[ylmkey] / dis;
                        valx = (d1 * delx + fn[nlmtp.n_id] * ylm_dx[ylmkey]);
                        valy = (d1 * dely + fn[nlmtp.n_id] * ylm_dy[ylmkey]);
                        valz = (d1 * delz + fn[nlmtp.n_id] * ylm_dz[ylmkey]);

                        anlm_dfx[nlmtp.nlmtp_key][atom1] += valx;
                        anlm_dfy[nlmtp.nlmtp_key][atom1] += valy;
                        anlm_dfz[nlmtp.nlmtp_key][atom1] += valz;
                        anlm_dfx[nlmtp.nlmtp_key][atom2] -= valx;
                        anlm_dfy[nlmtp.nlmtp_key][atom2] -= valy;
                        anlm_dfz[nlmtp.nlmtp_key][atom2] -= valz;
                        anlm_ds[nlmtp.nlmtp_key][0] -= valx * delx;
                        anlm_ds[nlmtp.nlmtp_key][1] -= valy * dely;
                        anlm_ds[nlmtp.nlmtp_key][2] -= valz * delz;
                        anlm_ds[nlmtp.nlmtp_key][3] -= valx * dely;
                        anlm_ds[nlmtp.nlmtp_key][4] -= valy * delz;
                        anlm_ds[nlmtp.nlmtp_key][5] -= valz * delx;
                    }
                }
            }
        }
    }

    int conj_key, nlmtp_key;
    for (const auto& nlmtp: nlmtp_attrs_no_conj){
        const auto& cc = nlmtp.lm.cc_coeff;
        conj_key = nlmtp.conj_key, nlmtp_key = nlmtp.nlmtp_key;
        anlm[conj_key] = cc * std::conj(anlm[nlmtp_key]);
        for (int k = 0; k < n_atom; ++k){
            anlm_dfx[conj_key][k] = cc * std::conj(anlm_dfx[nlmtp_key][k]);
            anlm_dfy[conj_key][k] = cc * std::conj(anlm_dfy[nlmtp_key][k]);
            anlm_dfz[conj_key][k] = cc * std::conj(anlm_dfz[nlmtp_key][k]);
        }
        for (int k = 0; k < 6; ++k){
            anlm_ds[conj_key][k] = cc * std::conj(anlm_ds[nlmtp_key][k]);
        }
    }
}
