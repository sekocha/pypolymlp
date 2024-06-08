/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/local.h"

Local::Local(){}

Local::Local(const int& n_atom_i,
             const int& atom1_i,
             const int& type1_i,
             const struct feature_params& fp_i,
             const ModelParams& modelp_i){

    n_atom = n_atom_i, atom1 = atom1_i, type1 = type1_i,
    fp = fp_i, modelp = modelp_i;

    n_type = fp.n_type;
    n_fn = modelp.get_n_fn(), n_des = modelp.get_n_des();

}

Local::~Local(){}

vector1d Local::pair(const vector2d& dis_a){

    vector1d an(n_des, 0.0);

    int col;
    double dis;
    vector1d fn;

    int n_type_comb(0);
    for (const auto& tc: modelp.get_type_comb_pair()){
        if (tc[type1].size() > 0){
            const int type2 = tc[type1][0];
            for (size_t j = 0; j < dis_a[type2].size(); ++j){
                dis = dis_a[type2][j];
                get_fn_(dis, fp, fn);
                for (int n = 0; n < n_fn; ++n) {
                    col = n_type_comb * n_fn + n;
                    an[col] += fn[n];
                }
            }
        }
        ++n_type_comb;
    }
    return an;
}

void Local::pair_d(const vector2d& dis_a,
                   const vector3d& diff_a,
                   const vector2i& atom2_a,
                   vector1d& an,
                   vector2d& an_dfx,
                   vector2d& an_dfy,
                   vector2d& an_dfz,
                   vector2d& an_ds){

    an = vector1d(n_des, 0.0);
    an_dfx = an_dfy = an_dfz = vector2d(n_des, vector1d(n_atom, 0.0));
    an_ds = vector2d(n_des, vector1d(6, 0.0));

    int atom2,col;
    double dis,delx,dely,delz,valx,valy,valz;
    vector1d fn,fn_d;

    int n_type_comb(0);
    for (const auto& tc: modelp.get_type_comb_pair()){
        if (tc[type1].size() > 0){
            const int type2 = tc[type1][0];
            for (size_t j = 0; j < dis_a[type2].size(); ++j){
                dis = dis_a[type2][j];
                delx = diff_a[type2][j][0];
                dely = diff_a[type2][j][1];
                delz = diff_a[type2][j][2];
                atom2 = atom2_a[type2][j];
                get_fn_(dis, fp, fn, fn_d);
                for (int n = 0; n < n_fn; ++n){
                    col = n_type_comb * n_fn + n;
                    an[col] += fn[n];
                    valx = fn_d[n] * delx / dis;
                    valy = fn_d[n] * dely / dis;
                    valz = fn_d[n] * delz / dis;
                    an_dfx[col][atom1] += valx;
                    an_dfy[col][atom1] += valy;
                    an_dfz[col][atom1] += valz;
                    an_dfx[col][atom2] -= valx;
                    an_dfy[col][atom2] -= valy;
                    an_dfz[col][atom2] -= valz;
                    an_ds[col][0] -= valx * delx;
                    an_ds[col][1] -= valy * dely;
                    an_ds[col][2] -= valz * delz;
                    an_ds[col][3] -= valx * dely;
                    an_ds[col][4] -= valy * delz;
                    an_ds[col][5] -= valz * delx;
                }
            }
        }
        ++n_type_comb;
    }
}

vector1d Local::gtinv(const vector2d& dis_a, const vector3d& diff_a){

    const vector2i &lm_info = get_lm_info(fp.maxl);
    const vector3dc &anlm = compute_anlm(dis_a, diff_a, lm_info);

    vector1d dn(n_des, 0.0);

    int nl(0), n_prod;
    dc prod_all;

    for (int n = 0; n < n_fn; ++n){
        const vector2dc &an = anlm[n];
        for (const auto& lin: modelp.get_linear_term_gtinv()){
            auto it = std::find(lin.type1.begin(), lin.type1.end(), type1);
            if (it != lin.type1.end()){
                const auto& lm_array = fp.lm_array[lin.lmindex];
                const auto& coeffs = fp.lm_coeffs[lin.lmindex];
                const auto& tc =
                    modelp.get_type_comb_pair(lin.tcomb_index, type1);
                n_prod = lm_array[0].size();
                for (size_t j = 0; j < lm_array.size(); ++j){
                    const vector1i &lm = lm_array[j];
                    prod_all = coeffs[j];
                    for (int s1 = 0; s1 < n_prod; ++s1){
                        prod_all *= an[tc[s1]][lm[s1]];
                    }
                    dn[nl] += std::real(prod_all);
                }
            }
            ++nl;
        }
    }
    return dn;
}


void Local::gtinv_d(const vector2d& dis_a,
                    const vector3d& diff_a,
                    const vector2i& atom2_a,
                    vector1d& dn,
                    vector2d& dn_dfx,
                    vector2d& dn_dfy,
                    vector2d& dn_dfz,
                    vector2d& dn_ds){

    const vector2i &lm_info = get_lm_info(fp.maxl);
    vector3dc anlm;
    vector4dc anlm_dfx, anlm_dfy, anlm_dfz, anlm_ds;
    compute_anlm_d(dis_a,
                   diff_a,
                   atom2_a,
                   lm_info,
                   anlm,
                   anlm_dfx,
                   anlm_dfy,
                   anlm_dfz,
                   anlm_ds);

    dn = vector1d(n_des, 0.0);
    dn_dfx = dn_dfy = dn_dfz = vector2d(n_des, vector1d(n_atom, 0.0));
    dn_ds = vector2d(n_des, vector1d(6, 0.0));

    int nl(0), n_prod, t1, lm1;
    double prod_r,prod_i,sum,sum0,sum1,sum2,sum3,sum4,sum5;
    dc prod, prod_all;
    vector1d sumx,sumy,sumz;
    for (int n = 0; n < n_fn; ++n){
        const vector2dc &an = anlm[n];
        const vector3dc &afxn = anlm_dfx[n];
        const vector3dc &afyn = anlm_dfy[n];
        const vector3dc &afzn = anlm_dfz[n];
        const vector3dc &asn = anlm_ds[n];
        for (const auto& lin: modelp.get_linear_term_gtinv()){
            auto it = std::find(lin.type1.begin(), lin.type1.end(), type1);
            if (it != lin.type1.end()){
                const auto& lm_array = fp.lm_array[lin.lmindex];
                const auto& coeffs = fp.lm_coeffs[lin.lmindex];
                const auto& tc =
                    modelp.get_type_comb_pair(lin.tcomb_index, type1);
                n_prod = lm_array[0].size();

                sum = sum0 = sum1 = sum2 = sum3 = sum4 = sum5 = 0.0;
                sumx = sumy = sumz = vector1d(n_atom, 0.0);
                for (size_t j = 0; j < lm_array.size(); ++j){
                    const vector1i &lm = lm_array[j];
                    prod_all = coeffs[j];
                    for (int s1 = 0; s1 < n_prod; ++s1){
                        t1 = tc[s1], lm1 = lm[s1];
                        const vector1dc &afx = afxn[t1][lm1];
                        const vector1dc &afy = afyn[t1][lm1];
                        const vector1dc &afz = afzn[t1][lm1];
                        const vector1dc &as = asn[t1][lm1];
                        prod_all *= an[t1][lm1];
                        prod = coeffs[j];
                        for (int s2 = 0; s2 < n_prod; ++s2){
                            if (s2 != s1) prod *= an[tc[s2]][lm[s2]];
                        }
                        prod_r = std::real(prod), prod_i = std::imag(prod);
                        for (int k = 0; k < n_atom; ++k){
                            sumx[k] += prod_r * afx[k].real()
                                - prod_i * afx[k].imag();
                            sumy[k] += prod_r * afy[k].real()
                                - prod_i * afy[k].imag();
                            sumz[k] += prod_r * afz[k].real()
                                - prod_i * afz[k].imag();
                        }
                        sum0 += prod_r * as[0].real() - prod_i * as[0].imag();
                        sum1 += prod_r * as[1].real() - prod_i * as[1].imag();
                        sum2 += prod_r * as[2].real() - prod_i * as[2].imag();
                        sum3 += prod_r * as[3].real() - prod_i * as[3].imag();
                        sum4 += prod_r * as[4].real() - prod_i * as[4].imag();
                        sum5 += prod_r * as[5].real() - prod_i * as[5].imag();
                    }
                    sum += std::real(prod_all);
                }
                dn[nl] = sum;
                for (int k = 0; k < n_atom; ++k) dn_dfx[nl][k] = sumx[k];
                for (int k = 0; k < n_atom; ++k) dn_dfy[nl][k] = sumy[k];
                for (int k = 0; k < n_atom; ++k) dn_dfz[nl][k] = sumz[k];
                dn_ds[nl][0] = sum0, dn_ds[nl][1] = sum1, dn_ds[nl][2] = sum2;
                dn_ds[nl][3] = sum3, dn_ds[nl][4] = sum4, dn_ds[nl][5] = sum5;
            }
            ++nl;
        }
    }
}

vector3dc Local::compute_anlm(const vector2d& dis_a,
                              const vector3d& diff_a,
                              const vector2i& lm_info){

    const int n_lm = lm_info.size(), n_lm_all = 2 * n_lm - fp.maxl - 1;

    vector3dc anlm(n_fn, vector2dc(n_type, vector1dc(n_lm_all, 0.0)));

    double dis,cc;
    vector1d fn;
    vector1dc ylm;

    for (int type2 = 0; type2 < n_type; ++type2){
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            const vector1d &sph = cartesian_to_spherical_(diff_a[type2][j]);
            get_fn_(dis, fp, fn);
            get_ylm_(sph[0], sph[1], fp.maxl, ylm);
            for (int n = 0; n < n_fn; ++n) {
                for (int lm = 0; lm < n_lm; ++lm) {
                    anlm[n][type2][lm_info[lm][2]] += fn[n] * ylm[lm];
                }
            }
        }
    }
    for (int n = 0; n < n_fn; ++n) {
        for (int type2 = 0; type2 < n_type; ++type2) {
            for (int lm = 0; lm < n_lm; ++lm) {
                cc = pow(-1, lm_info[lm][1]);
                anlm[n][type2][lm_info[lm][3]]
                    = cc * std::conj(anlm[n][type2][lm_info[lm][2]]);
            }
        }
    }
    return anlm;
}

void Local::compute_anlm_d(const vector2d& dis_a,
                           const vector3d& diff_a,
                           const vector2i& atom2_a,
                           const vector2i& lm_info,
                           vector3dc& anlm,
                           vector4dc& anlm_dfx,
                           vector4dc& anlm_dfy,
                           vector4dc& anlm_dfz,
                           vector4dc& anlm_ds){

    const int n_lm = lm_info.size(), n_lm_all = 2 * n_lm - fp.maxl - 1;

    anlm = vector3dc(n_fn, vector2dc(n_type, vector1dc(n_lm_all, 0.0)));
    anlm_dfx = anlm_dfy = anlm_dfz = vector4dc
        (n_fn, vector3dc(n_type, vector2dc(n_lm_all, vector1dc(n_atom, 0.0))));
    anlm_ds = vector4dc
        (n_fn, vector3dc(n_type, vector2dc(n_lm_all, vector1dc(6, 0.0))));

    vector1d fn,fn_d;
    vector1dc ylm,ylm_dx,ylm_dy,ylm_dz;
    double delx,dely,delz,dis,cc;
    dc d1,valx,valy,valz;
    int atom2,m,lm1,lm2;

    for (int type2 = 0; type2 < n_type; ++type2){
        for (size_t j = 0; j < dis_a[type2].size(); ++j){
            dis = dis_a[type2][j];
            delx = diff_a[type2][j][0];
            dely = diff_a[type2][j][1];
            delz = diff_a[type2][j][2];
            atom2 = atom2_a[type2][j];
            const vector1d &sph = cartesian_to_spherical_(diff_a[type2][j]);
            get_fn_(dis, fp, fn, fn_d);
            get_ylm_(dis, sph[0], sph[1], fp.maxl,
                     ylm, ylm_dx, ylm_dy, ylm_dz);

            for (int lm = 0; lm < n_lm; ++lm) {
                lm1 = lm_info[lm][2], lm2 = lm_info[lm][3];
                for (int n = 0; n < n_fn; ++n) {
                    anlm[n][type2][lm1] += fn[n] * ylm[lm];
                    d1 = fn_d[n] * ylm[lm] / dis;
                    valx = (d1 * delx + fn[n] * ylm_dx[lm]);
                    valy = (d1 * dely + fn[n] * ylm_dy[lm]);
                    valz = (d1 * delz + fn[n] * ylm_dz[lm]);
                    anlm_dfx[n][type2][lm1][atom1] += valx;
                    anlm_dfy[n][type2][lm1][atom1] += valy;
                    anlm_dfz[n][type2][lm1][atom1] += valz;
                    anlm_dfx[n][type2][lm1][atom2] -= valx;
                    anlm_dfy[n][type2][lm1][atom2] -= valy;
                    anlm_dfz[n][type2][lm1][atom2] -= valz;
                    anlm_ds[n][type2][lm1][0] -= valx * delx;
                    anlm_ds[n][type2][lm1][1] -= valy * dely;
                    anlm_ds[n][type2][lm1][2] -= valz * delz;
                    anlm_ds[n][type2][lm1][3] -= valx * dely;
                    anlm_ds[n][type2][lm1][4] -= valy * delz;
                    anlm_ds[n][type2][lm1][5] -= valz * delx;
                }
            }
        }
    }

    for (int n = 0; n < n_fn; ++n) {
        for (int type2 = 0; type2 < n_type; ++type2) {
            for (int lm = 0; lm < n_lm; ++lm) {
                m = lm_info[lm][1], lm1 = lm_info[lm][2], lm2 = lm_info[lm][3];
                cc = pow(-1, m);
                anlm[n][type2][lm2] = cc * std::conj(anlm[n][type2][lm1]);
                for (int k = 0; k < n_atom; ++k){
                    anlm_dfx[n][type2][lm2][k]
                        = cc * std::conj(anlm_dfx[n][type2][lm1][k]);
                    anlm_dfy[n][type2][lm2][k]
                        = cc * std::conj(anlm_dfy[n][type2][lm1][k]);
                    anlm_dfz[n][type2][lm2][k]
                        = cc * std::conj(anlm_dfz[n][type2][lm1][k]);
                }
                for (int k = 0; k < 6; ++k){
                    anlm_ds[n][type2][lm2][k]
                        = cc * std::conj(anlm_ds[n][type2][lm1][k]);
                }
            }
        }
    }
}

vector2i Local::get_lm_info(const int& max_l){

    vector2i lm_comb;
    for (int l = 0; l < max_l + 1; ++l){
        for (int m = -l; m < 1; ++m){
            lm_comb.emplace_back(vector1i{l,m,l*l+l+m,l*l+l-m});
        }
    }
    return lm_comb;
}
