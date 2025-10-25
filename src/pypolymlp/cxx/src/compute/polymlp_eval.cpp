/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_eval.h"


PolymlpEval::PolymlpEval(){}

PolymlpEval::PolymlpEval(const feature_params& fp, const vector1d& coeffs){

    vector1d coeffs_rev(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) coeffs_rev[i] = 2.0 * coeffs[i];
    polymlp_api.set_potential_model(fp, coeffs_rev);
}

PolymlpEval::~PolymlpEval(){}


void PolymlpEval::eval(
    const vector1i& types,
    const vector2i& neighbor_half,
    const vector3d& neighbor_diff,
    double& energy,
    vector2d& forces,
    vector1d& stress
){

    const auto& fp = polymlp_api.get_fp();
    if (fp.feature_type == "pair"){
        eval_pair(types, neighbor_half, neighbor_diff, energy, forces, stress);
    }
    else if (fp.feature_type == "gtinv"){
        eval_gtinv(types, neighbor_half, neighbor_diff, energy, forces, stress);
    }
}


void PolymlpEval::eval_pair(
    const vector1i& types,
    const vector2i& neighbor_half,
    const vector3d& neighbor_diff,
    double& energy,
    vector2d& forces,
    vector1d& stress
){

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_atom = types.size();
    vector2d antp, prod_sum_e, prod_sum_f;
    compute_antp(types, neighbor_half, neighbor_diff, antp);
    compute_sum_of_prod_antp(types, antp, prod_sum_e, prod_sum_f);

    energy = 0.0;
    forces.resize(n_atom);
    for (int i = 0; i < n_atom; ++i) forces[i] = vector1d(3, 0.0);
    stress = vector1d(6, 0.0);

    int type1, type2, tp;
    double dx, dy, dz, dis, e_ij, f_ij, fx, fy, fz;
    vector1d fn, fn_d;
    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const vector1i& neighbor_i = neighbor_half[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;

        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            type2 = types[j];
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn, fn_d);
                e_ij = 0.0, f_ij = 0.0;
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        const int idx_i = ntp.ilocal_id;
                        const int idx_j = ntp.jlocal_id;
                        const auto& prod_ei = prod_sum_e[i][idx_i];
                        const auto& prod_ej = prod_sum_e[j][idx_j];
                        const auto& prod_fi = prod_sum_f[i][idx_i];
                        const auto& prod_fj = prod_sum_f[j][idx_j];
                        e_ij += fn[ntp.n_id] * (prod_ei + prod_ej);
                        f_ij += fn_d[ntp.n_id] * (prod_fi + prod_fj);
                    }
                }
                f_ij *= - 1.0 / dis;
                fx = f_ij * dx;
                fy = f_ij * dy;
                fz = f_ij * dz;

                energy += e_ij;
                forces[i][0] += fx, forces[i][1] += fy, forces[i][2] += fz;
                forces[j][0] -= fx, forces[j][1] -= fy, forces[j][2] -= fz;
                stress[0] += dx * fx;
                stress[1] += dy * fy;
                stress[2] += dz * fz;
                stress[3] += dx * fy;
                stress[4] += dy * fz;
                stress[5] += dz * fx;
            }
        }
    }
}

void PolymlpEval::compute_antp(
    const vector1i& types,
    const vector2i& neighbor_half,
    const vector3d& neighbor_diff,
    vector2d& antp
){

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_atom = types.size();
    antp = vector2d(n_atom);
    for (int i = 0; i < n_atom; ++i) {
        int type1 = types[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;
        antp[i] = vector1d(ntp_attrs.size(), 0.0);
    }

    int type1, type2, tp;
    double dx, dy, dz, dis;
    vector1d fn;
    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const vector1i& neighbor_i = neighbor_half[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;

        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < fp.cutoff){
                type2 = types[j];
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn);
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        const int idx_i = ntp.ilocal_id;
                        const int idx_j = ntp.jlocal_id;
                        antp[i][idx_i] += fn[ntp.n_id];
                        antp[j][idx_j] += fn[ntp.n_id];
                    }
                }
            }
        }
    }
}

void PolymlpEval::compute_sum_of_prod_antp(
    const vector1i& types,
    const vector2d& antp,
    vector2d& prod_sum_e,
    vector2d& prod_sum_f
){

    const int n_atom = types.size();
    prod_sum_e = vector2d(n_atom);
    prod_sum_f = vector2d(n_atom);

    for (int i = 0; i < n_atom; ++i) {
        const int type1 = types[i];
        polymlp_api.compute_sum_of_prod_antp(
            antp[i], type1, prod_sum_e[i], prod_sum_f[i]
        );
    }
}


/*--- feature_type = gtinv ----------------------------------------------*/
void PolymlpEval::eval_gtinv(
    const vector1i& types,
    const vector2i& neighbor_half,
    const vector3d& neighbor_diff,
    double& energy,
    vector2d& forces,
    vector1d& stress
){

    const int n_atom = types.size();
    vector2dc anlmtp, prod_sum_e, prod_sum_f;
    compute_anlmtp(types, neighbor_half, neighbor_diff, anlmtp);
    compute_sum_of_prod_anlmtp(types, anlmtp, prod_sum_e, prod_sum_f);

    energy = 0.0;
    forces.resize(n_atom);
    for (int i = 0; i < n_atom; ++i) forces[i] = vector1d(3, 0.0);
    stress = vector1d(6, 0.0);

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    int type1, type2, tp;
    double dx, dy, dz, dis, e_ij, fx, fy, fz;
    dc val, valx, valy, valz, d1;
    vector1d fn, fn_d;
    vector1dc ylm, ylm_dx, ylm_dy, ylm_dz;

    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const vector1i& neighbor_i = neighbor_half[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;
        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            type2 = types[j];
            // diff = pos[j] - pos[i], (dx, dy, dz) = pos[i] - pos[j]
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                const auto& sph = cartesian_to_spherical_(vector1d{dx,dy,dz});
                get_fn_(dis, fp, params, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], fp.maxl,
                         ylm, ylm_dx, ylm_dy, ylm_dz);

                e_ij = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                for (const auto& nlmtp: nlmtp_attrs_noconj){
                    if (tp == nlmtp.tp){
                        const auto& lm_attr = nlmtp.lm;
                        const int ylmkey = lm_attr.ylmkey;
                        const int idx_i = nlmtp.ilocal_noconj_id;
                        const int idx_j = nlmtp.jlocal_noconj_id;
                        val = fn[nlmtp.n_id] * ylm[ylmkey];
                        d1 = fn_d[nlmtp.n_id] * ylm[ylmkey] / dis;
                        valx = - (d1 * dx + fn[nlmtp.n_id] * ylm_dx[ylmkey]);
                        valy = - (d1 * dy + fn[nlmtp.n_id] * ylm_dy[ylmkey]);
                        valz = - (d1 * dz + fn[nlmtp.n_id] * ylm_dz[ylmkey]);
                        const auto& prod_ei = prod_sum_e[i][idx_i];
                        const auto& prod_ej = prod_sum_e[j][idx_j];
                        const auto& prod_fi = prod_sum_f[i][idx_i];
                        const auto& prod_fj = prod_sum_f[j][idx_j];
                        const dc sum_e = prod_ei + prod_ej * lm_attr.sign_j;
                        const dc sum_f = prod_fi + prod_fj * lm_attr.sign_j;
                        if (lm_attr.m == 0){
                            e_ij += 0.5 * prod_real(val, sum_e);
                            fx += 0.5 * prod_real(valx, sum_f);
                            fy += 0.5 * prod_real(valy, sum_f);
                            fz += 0.5 * prod_real(valz, sum_f);
                        }
                        else {
                            e_ij += prod_real(val, sum_e);
                            fx += prod_real(valx, sum_f);
                            fy += prod_real(valy, sum_f);
                            fz += prod_real(valz, sum_f);
                        }
                    }
                }
                energy += e_ij;
                forces[i][0] += fx, forces[i][1] += fy, forces[i][2] += fz;
                forces[j][0] -= fx, forces[j][1] -= fy, forces[j][2] -= fz;
                stress[0] += dx * fx;
                stress[1] += dy * fy;
                stress[2] += dz * fz;
                stress[3] += dx * fy;
                stress[4] += dy * fz;
                stress[5] += dz * fx;
            }
        }
    }
}


void PolymlpEval::compute_anlmtp(
    const vector1i& types,
    const vector2i& neighbor_half,
    const vector3d& neighbor_diff,
    vector2dc& anlmtp
){

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    const int n_atom = types.size();
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
        const vector1i& neighbor_i = neighbor_half[i];
        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            const auto& maps_type = maps.maps_type[type1];
            const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;

            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < fp.cutoff){
                type2 = types[j];
                const auto &sph = cartesian_to_spherical_(vector1d{dx,dy,dz});
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, fp, params, fn);
                get_ylm_(sph[0], sph[1], fp.maxl, ylm);
                for (const auto& nlmtp: nlmtp_attrs_noconj){
                    if (tp == nlmtp.tp){
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
        }
    }
    compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, types, anlmtp);
}


void PolymlpEval::compute_anlmtp_conjugate(
    const vector2d& anlmtp_r,
    const vector2d& anlmtp_i,
    const vector1i& types,
    vector2dc& anlmtp
){
    const int n_atom = types.size();
    anlmtp = vector2dc(n_atom);
    for (int i = 0; i < n_atom; ++i) {
        polymlp_api.compute_anlmtp_conjugate(
            anlmtp_r[i], anlmtp_i[i], types[i], anlmtp[i]);
    }
}


void PolymlpEval::compute_sum_of_prod_anlmtp(
    const vector1i& types,
    const vector2dc& anlmtp,
    vector2dc& prod_sum_e,
    vector2dc& prod_sum_f
){

    const int n_atom = types.size();
    prod_sum_e = vector2dc(n_atom);
    prod_sum_f = vector2dc(n_atom);
    for (int i = 0; i < n_atom; ++i) {
        polymlp_api.compute_sum_of_prod_anlmtp(
            anlmtp[i], types[i], prod_sum_e[i], prod_sum_f[i]
        );
    }
}
