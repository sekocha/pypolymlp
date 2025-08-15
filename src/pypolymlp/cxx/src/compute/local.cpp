/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/local.h"

#include <chrono>

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
    auto t1 = std::chrono::system_clock::now();
    compute_anlmtp_d(
        polymlp, atom1, type1, dis_a, diff_a, atom2_a,
        anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds
    );

    auto t2 = std::chrono::system_clock::now();
    polymlp.compute_features(anlmtp, type1, dn);
    auto t3 = std::chrono::system_clock::now();
    polymlp.compute_features_deriv(
        anlmtp, anlmtp_dfx, anlmtp_dfy, anlmtp_dfz, anlmtp_ds,
        type1, dn_dfx, dn_dfy, dn_dfz, dn_ds
    );
    /*
    auto t4 = std::chrono::system_clock::now();
    double time;
    time = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>
            (t2 - t1).count() / 1000.0
        );
    std::cout << "anlmtp_d:" << time << std::endl;
    time = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>
            (t3 - t2).count() / 1000.0
        );
    std::cout << "features:" << time << std::endl;
    time = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>
            (t4 - t3).count() / 1000.0
        );
    std::cout << "features_deriv:" << time << std::endl;
    */
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
                    if (tp == nlmtp.tp and fabs(fn[nlmtp.n_id]) > tol){
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

                        auto& xr = anlmtp_dfx_r[idx_i];
                        xr[atom1] += valx.real();
                        xr[atom2] -= valx.real();

                        auto& yr = anlmtp_dfy_r[idx_i];
                        yr[atom1] += valy.real();
                        yr[atom2] -= valy.real();

                        auto& zr = anlmtp_dfz_r[idx_i];
                        zr[atom1] += valz.real();
                        zr[atom2] -= valz.real();

                        auto& xi = anlmtp_dfx_i[idx_i];
                        xi[atom1] += valx.imag();
                        xi[atom2] -= valx.imag();

                        auto& yi = anlmtp_dfy_i[idx_i];
                        yi[atom1] += valy.imag();
                        yi[atom2] -= valy.imag();

                        auto& zi = anlmtp_dfz_i[idx_i];
                        zi[atom1] += valz.imag();
                        zi[atom2] -= valz.imag();

                        /*
                        anlmtp_dfx_r[idx_i][atom1] += valx.real();
                        anlmtp_dfx_r[idx_i][atom2] -= valx.real();

                        anlmtp_dfy_r[idx_i][atom1] += valy.real();
                        anlmtp_dfy_r[idx_i][atom2] -= valy.real();

                        anlmtp_dfz_r[idx_i][atom1] += valz.real();
                        anlmtp_dfz_r[idx_i][atom2] -= valz.real();

                        anlmtp_dfx_i[idx_i][atom1] += valx.imag();
                        anlmtp_dfx_i[idx_i][atom2] -= valx.imag();

                        anlmtp_dfy_i[idx_i][atom1] += valy.imag();
                        anlmtp_dfy_i[idx_i][atom2] -= valy.imag();

                        anlmtp_dfz_i[idx_i][atom1] += valz.imag();
                        anlmtp_dfz_i[idx_i][atom2] -= valz.imag();
                        */

                        val0 = valx * delx;
                        anlmtp_ds_r[idx_i][0] -= val0.real();
                        anlmtp_ds_i[idx_i][0] -= val0.imag();
                        val1 = valy * dely;
                        anlmtp_ds_r[idx_i][1] -= val1.real();
                        anlmtp_ds_i[idx_i][1] -= val1.imag();
                        val2 = valz * delz;
                        anlmtp_ds_r[idx_i][2] -= val2.real();
                        anlmtp_ds_i[idx_i][2] -= val2.imag();
                        val3 = valx * dely;
                        anlmtp_ds_r[idx_i][3] -= val3.real();
                        anlmtp_ds_i[idx_i][3] -= val3.imag();
                        val4 = valy * delz;
                        anlmtp_ds_r[idx_i][4] -= val4.real();
                        anlmtp_ds_i[idx_i][4] -= val4.imag();
                        val5 = valz * delx;
                        anlmtp_ds_r[idx_i][5] -= val5.real();
                        anlmtp_ds_i[idx_i][5] -= val5.imag();
                    }
                }
            }
        }
    }
    compute_conjugate(
        polymlp,
        type1,
        anlmtp_r,
        anlmtp_i,
        anlmtp_dfx_r,
        anlmtp_dfx_i,
        anlmtp_dfy_r,
        anlmtp_dfy_i,
        anlmtp_dfz_r,
        anlmtp_dfz_i,
        anlmtp_ds_r,
        anlmtp_ds_i,
        anlmtp,
        anlmtp_dfx,
        anlmtp_dfy,
        anlmtp_dfz,
        anlmtp_ds
    );
/*
    polymlp.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, type1, anlmtp);
    polymlp.compute_anlmtp_conjugate(anlmtp_dfx_r, anlmtp_dfx_i, type1, anlmtp_dfx);
    polymlp.compute_anlmtp_conjugate(anlmtp_dfy_r, anlmtp_dfy_i, type1, anlmtp_dfy);
    polymlp.compute_anlmtp_conjugate(anlmtp_dfz_r, anlmtp_dfz_i, type1, anlmtp_dfz);
    polymlp.compute_anlmtp_conjugate(anlmtp_ds_r, anlmtp_ds_i, type1, anlmtp_ds);
*/
}


void Local::compute_conjugate(
    PolymlpAPI& polymlp,
    const int type1,
    const vector1d& anlmtp_r,
    const vector1d& anlmtp_i,
    const vector2d& anlmtp_dfx_r,
    const vector2d& anlmtp_dfx_i,
    const vector2d& anlmtp_dfy_r,
    const vector2d& anlmtp_dfy_i,
    const vector2d& anlmtp_dfz_r,
    const vector2d& anlmtp_dfz_i,
    const vector2d& anlmtp_ds_r,
    const vector2d& anlmtp_ds_i,
    vector1dc& anlmtp,
    vector2dc& anlmtp_dfx,
    vector2dc& anlmtp_dfy,
    vector2dc& anlmtp_dfz,
    vector2dc& anlmtp_ds
){
    const auto& maps = polymlp.get_maps();
    const auto& maps_type = maps.maps_type[type1];
    const auto& nlmtp_attrs = maps_type.nlmtp_attrs;
    const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

    const int n_atom = anlmtp_dfx_r[0].size();
    anlmtp = vector1dc(nlmtp_attrs.size(), 0.0);
    anlmtp_dfx = vector2dc(nlmtp_attrs.size(), vector1dc(n_atom, 0.0));
    anlmtp_dfy = vector2dc(nlmtp_attrs.size(), vector1dc(n_atom, 0.0));
    anlmtp_dfz = vector2dc(nlmtp_attrs.size(), vector1dc(n_atom, 0.0));
    anlmtp_ds = vector2dc(nlmtp_attrs.size(), vector1dc(6, 0.0));

    int idx(0);
    double rval, ival;
    for (const auto& nlmtp: nlmtp_attrs_noconj){
        const auto& cc_coeff = nlmtp.lm.cc_coeff;
        const int id1 = nlmtp.ilocal_id;
        const int id2 = nlmtp.ilocal_conj_id;
        anlmtp[id1] = {anlmtp_r[idx], anlmtp_i[idx]};
        anlmtp[id2] = {cc_coeff * anlmtp_r[idx], - cc_coeff * anlmtp_i[idx]};

        for (size_t k = 0; k < n_atom; ++k){
            rval = anlmtp_dfx_r[idx][k];
            ival = anlmtp_dfx_i[idx][k];
            anlmtp_dfx[id1][k] = {rval, ival};
            anlmtp_dfx[id2][k] = {cc_coeff * rval, - cc_coeff * ival};

            rval = anlmtp_dfy_r[idx][k];
            ival = anlmtp_dfy_i[idx][k];
            anlmtp_dfy[id1][k] = {rval, ival};
            anlmtp_dfy[id2][k] = {cc_coeff * rval, - cc_coeff * ival};

            rval = anlmtp_dfz_r[idx][k];
            ival = anlmtp_dfz_i[idx][k];
            anlmtp_dfz[id1][k] = {rval, ival};
            anlmtp_dfz[id2][k] = {cc_coeff * rval, - cc_coeff * ival};
        }
        for (size_t k = 0; k < 6; ++k){
            rval = anlmtp_ds_r[idx][k];
            ival = anlmtp_ds_i[idx][k];
            anlmtp_ds[id1][k] = {rval, ival};
            anlmtp_ds[id2][k] = {cc_coeff * rval, - cc_coeff * ival};
        }
        ++idx;
    }

}
