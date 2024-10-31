/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_eval.h"


PolymlpEval::PolymlpEval(){}

PolymlpEval::PolymlpEval(const feature_params& fp, const vector1d& coeffs){

    vector1d coeffs_rev(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) coeffs_rev[i] = 2.0 * coeffs[i];

    const Features f_obj(fp);
    pot.fp = fp;
    pot.mapping = f_obj.get_mapping();
    pot.modelp = f_obj.get_model_params();
    pot.p_obj = Potential(f_obj, coeffs_rev);

    type_pairs = pot.mapping.get_type_pairs();

}

PolymlpEval::~PolymlpEval(){}

void PolymlpEval::eval(const vector2d& positions_c,
                       const vector1i& types,
                       const vector2i& neighbor_half,
                       const vector3d& neighbor_diff,
                       double& energy,
                       vector2d& forces,
                       vector1d& stress){

    if (pot.fp.feature_type == "pair"){
        eval_pair(positions_c, types, neighbor_half, neighbor_diff,
                  energy, forces, stress);
    }
    else if (pot.fp.feature_type == "gtinv"){
        eval_gtinv(positions_c, types, neighbor_half, neighbor_diff,
                   energy, forces, stress);
    }

}

/*--- feature_type = pair ----------------------------------------------*/

void PolymlpEval::eval_pair(const vector2d& positions_c,
                            const vector1i& types,
                            const vector2i& neighbor_half,
                            const vector3d& neighbor_diff,
                            double& energy,
                            vector2d& forces,
                            vector1d& stress){

    const auto& ntp_attrs = pot.mapping.get_ntp_attrs();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    const int n_atom = types.size();
    vector2d antp, prod_sum_e, prod_sum_f;
    compute_antp(positions_c, types, neighbor_half, neighbor_diff, antp);
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
        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            type2 = types[j];
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < pot.fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn, fn_d);
                e_ij = 0.0, f_ij = 0.0;
                int head_key(0);
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        const auto& prod_ei = prod_sum_e[i][head_key];
                        const auto& prod_ej = prod_sum_e[j][head_key];
                        const auto& prod_fi = prod_sum_f[i][head_key];
                        const auto& prod_fj = prod_sum_f[j][head_key];
                        e_ij += fn[ntp.n_id] * (prod_ei + prod_ej);
                        f_ij += fn_d[ntp.n_id] * (prod_fi + prod_fj);
                    }
                    ++head_key;
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

void PolymlpEval::compute_antp(const vector2d& positions_c,
                               const vector1i& types,
                               const vector2i& neighbor_half,
                               const vector3d& neighbor_diff,
                               vector2d& antp){

    const int n_atom = types.size();
    const auto& ntp_attrs = pot.mapping.get_ntp_attrs();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    antp = vector2d(n_atom, vector1d(ntp_attrs.size(), 0.0));

    int type1, type2, tp;
    double dx, dy, dz, dis;
    vector1d fn;
    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const vector1i& neighbor_i = neighbor_half[i];
        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < pot.fp.cutoff){
                type2 = types[j];
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn);
                int idx(0);
                for (const auto& ntp: ntp_attrs){
                    if (tp == ntp.tp){
                        antp[i][idx] += fn[ntp.n_id];
                        antp[j][idx] += fn[ntp.n_id];
                    }
                    ++idx;
                }
            }
        }
    }
}

void PolymlpEval::compute_sum_of_prod_antp(const vector1i& types,
                                           const vector2d& antp,
                                           vector2d& prod_antp_sum_e,
                                           vector2d& prod_antp_sum_f){

    const auto& ntp_attrs = pot.mapping.get_ntp_attrs();

    const int n_atom = antp.size();
    prod_antp_sum_e = vector2d(n_atom, vector1d(ntp_attrs.size(), 0.0));
    prod_antp_sum_f = vector2d(n_atom, vector1d(ntp_attrs.size(), 0.0));

    for (int i = 0; i < n_atom; ++i) {
        int type1 = types[i];
        const auto& linear_features = pot.p_obj.get_linear_features(type1);
        const auto& prod_map = pot.p_obj.get_prod_map(type1);
        const auto& prod_features_map = pot.p_obj.get_prod_features_map(type1);

        // computing products of order parameters (antp)
        vector1d prod_antp;
        compute_products<double>(prod_map, antp[i], prod_antp);
        // end: computing products of order parameters (antp)

        // computing linear features
        vector1d feature_values(linear_features.size(), 0.0);
        int idx = 0;
        for (const auto& sfeature: linear_features){
            if (sfeature.size() > 0){
                feature_values[idx] = prod_antp[sfeature[0].prod_key];
            }
            ++idx;
        }
        // end: computing linear features

        vector1d prod_features;
        compute_products<double>(prod_features_map,
                                 feature_values,
                                 prod_features);

        idx = 0;
        for (const auto& ntp: ntp_attrs){
            const auto& pmodel = pot.p_obj.get_potential_model(type1,
                                                               ntp.ntp_key);
            double sum_e(0.0), sum_f(0.0), prod;
            for (const auto& pterm: pmodel){
                prod = prod_antp[pterm.prod_key]
                     * prod_features[pterm.prod_features_key];
                sum_e += pterm.coeff_e * prod;
                sum_f += pterm.coeff_f * prod;
            }
            prod_antp_sum_e[i][idx] = 0.5 * sum_e;
            prod_antp_sum_f[i][idx] = 0.5 * sum_f;
            ++idx;
        }
    }
}


/*--- feature_type = gtinv ----------------------------------------------*/
void PolymlpEval::eval_gtinv(const vector2d& positions_c,
                             const vector1i& types,
                             const vector2i& neighbor_half,
                             const vector3d& neighbor_diff,
                             double& energy,
                             vector2d& forces,
                             vector1d& stress){

    const int n_atom = types.size();

    vector2dc anlmtp, prod_sum_e, prod_sum_f;
    clock_t t1 = clock();
    compute_anlmtp(positions_c, types, neighbor_half, neighbor_diff, anlmtp);
    clock_t t2 = clock();
    compute_sum_of_prod_anlmtp(types, anlmtp, prod_sum_e, prod_sum_f);
    clock_t t3 = clock();

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    energy = 0.0;
    forces.resize(n_atom);
    for (int i = 0; i < n_atom; ++i) forces[i] = vector1d(3, 0.0);
    stress = vector1d(6, 0.0);

    int type1, type2, tp;
    double dx, dy, dz, dis;
    double e_ij, fx, fy, fz;
    dc val, valx, valy, valz, d1;
    vector1d fn, fn_d;
    vector1dc ylm, ylm_dx, ylm_dy, ylm_dz;

    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const vector1i& neighbor_i = neighbor_half[i];
        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            type2 = types[j];
            // diff = pos[j] - pos[i], (dx, dy, dz) = pos[i] - pos[j]
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < pot.fp.cutoff){
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                const auto& sph = cartesian_to_spherical_(vector1d{dx,dy,dz});
                get_fn_(dis, pot.fp, params, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], pot.fp.maxl,
                         ylm, ylm_dx, ylm_dy, ylm_dz);

                e_ij = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    const auto& lm_attr = nlmtp.lm;
                    const int ylmkey = lm_attr.ylmkey;
                    const int head_key = nlmtp.nlmtp_noconj_key;
                    if (tp == nlmtp.tp){
                        val = fn[nlmtp.n_id] * ylm[ylmkey];
                        d1 = fn_d[nlmtp.n_id] * ylm[ylmkey] / dis;
                        valx = - (d1 * dx + fn[nlmtp.n_id] * ylm_dx[ylmkey]);
                        valy = - (d1 * dy + fn[nlmtp.n_id] * ylm_dy[ylmkey]);
                        valz = - (d1 * dz + fn[nlmtp.n_id] * ylm_dz[ylmkey]);
                        const auto& prod_ei = prod_sum_e[i][head_key];
                        const auto& prod_ej = prod_sum_e[j][head_key];
                        const auto& prod_fi = prod_sum_f[i][head_key];
                        const auto& prod_fj = prod_sum_f[j][head_key];
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
    /*
    clock_t t4 = clock();
    std::cout << "all"
        << double(t2-t1)/CLOCKS_PER_SEC << " "
        << double(t3-t2)/CLOCKS_PER_SEC << " "
        << double(t4-t3)/CLOCKS_PER_SEC << " "
        << std::endl;
    */

}


void PolymlpEval::compute_anlmtp(const vector2d& positions_c,
                                 const vector1i& types,
                                 const vector2i& neighbor_half,
                                 const vector3d& neighbor_diff,
                                 vector2dc& anlmtp){

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& tp_to_params = pot.mapping.get_type_pair_to_params();

    const int n_atom = types.size();
    vector2d anlmtp_r(n_atom, vector1d(nlmtp_attrs_no_conj.size(), 0.0));
    vector2d anlmtp_i(n_atom, vector1d(nlmtp_attrs_no_conj.size(), 0.0));

    int type1, type2, tp;
    double dx, dy, dz, dis;
    vector1d fn; vector1dc ylm; dc val;
    for (int i = 0; i < n_atom; ++i) {
        type1 = types[i];
        const vector1i& neighbor_i = neighbor_half[i];
        for (size_t jj = 0; jj < neighbor_i.size(); ++jj){
            int j = neighbor_i[jj];
            const auto& diff = neighbor_diff[i][jj];
            dx = - diff[0];
            dy = - diff[1];
            dz = - diff[2];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis < pot.fp.cutoff){
                type2 = types[j];
                const auto &sph = cartesian_to_spherical_(vector1d{dx,dy,dz});
                tp = type_pairs[type1][type2];
                const auto& params = tp_to_params[tp];
                get_fn_(dis, pot.fp, params, fn);
                get_ylm_(sph[0], sph[1], pot.fp.maxl, ylm);
                for (const auto& nlmtp: nlmtp_attrs_no_conj){
                    const auto& lm_attr = nlmtp.lm;
                    const int idx = nlmtp.nlmtp_noconj_key;
                    if (tp == nlmtp.tp){
                        val = fn[nlmtp.n_id] * ylm[lm_attr.ylmkey];
                        anlmtp_r[i][idx] += val.real();
                        anlmtp_r[j][idx] += val.real() * lm_attr.sign_j;
                        anlmtp_i[i][idx] += val.imag();
                        anlmtp_i[j][idx] += val.imag() * lm_attr.sign_j;
                    }
                }
            }
        }
    }
    compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, anlmtp);
}


void PolymlpEval::compute_anlmtp_conjugate(const vector2d& anlmtp_r,
                                           const vector2d& anlmtp_i,
                                           vector2dc& anlmtp){

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const auto& n_nlmtp_all = pot.mapping.get_n_nlmtp_all();
    const int n_atom = anlmtp_r.size();
    anlmtp = vector2dc(n_atom, vector1dc(n_nlmtp_all, 0.0));

//    #ifdef _OPENMP
//    #pragma omp parallel for schedule(guided)
//    #endif
    for (int i = 0; i < n_atom; ++i) {
        int idx(0);
        for (const auto& nlmtp: nlmtp_attrs_no_conj){
            const auto& cc_coeff = nlmtp.lm.cc_coeff;
            anlmtp[i][nlmtp.nlmtp_key] = {anlmtp_r[i][idx], anlmtp_i[i][idx]};
            anlmtp[i][nlmtp.conj_key] = {cc_coeff * anlmtp_r[i][idx],
                                          - cc_coeff * anlmtp_i[i][idx]};
            ++idx;
        }
    }
}


void PolymlpEval::compute_sum_of_prod_anlmtp(const vector1i& types,
                                             const vector2dc& anlmtp,
                                             vector2dc& prod_sum_e,
                                             vector2dc& prod_sum_f){

    const auto& nlmtp_attrs_no_conj = pot.mapping.get_nlmtp_attrs_no_conjugate();
    const int n_head_keys = nlmtp_attrs_no_conj.size();
    const int n_atom = types.size();
    prod_sum_e = vector2dc(n_atom, vector1dc(n_head_keys));
    prod_sum_f = vector2dc(n_atom, vector1dc(n_head_keys));

    for (int i = 0; i < n_atom; ++i) {
        int type1 = types[i];

        const auto& prod_map = pot.p_obj.get_prod_map(type1);
        const auto& prod_map_erased = pot.p_obj.get_prod_map_erased(type1);
        const auto& prod_features_map = pot.p_obj.get_prod_features_map(type1);

        clock_t t1 = clock();
        // computing nonequivalent products of order parameters (anlmtp)
        vector1d prod_anlmtp;
        compute_products_real(prod_map, anlmtp[i], prod_anlmtp);
        clock_t t2 = clock();
        // end: computing products of order parameters (anlmtp)

        // computing linear features
        //   and nonequivalent products of linear features
        vector1d features, prod_features;
        compute_linear_features(prod_anlmtp, type1, features);
        compute_products<double>(prod_features_map, features, prod_features);
        // end: computing linear features
        clock_t t3 = clock();

        vector1dc prod_anlmtp_erased;
        compute_products<dc>(prod_map_erased, anlmtp[i], prod_anlmtp_erased);
        clock_t t4 = clock();

        for (size_t key = 0; key < nlmtp_attrs_no_conj.size(); ++key){
            const auto& pmodel = pot.p_obj.get_potential_model(type1, key);
            dc sum_e(0.0), sum_f(0.0);
            //dc prod;
            for (const auto& pterm: pmodel){
                // TODO: examine accuracy
                if (fabs(prod_features[pterm.prod_features_key]) > 1e-50){
                    sum_e += pterm.coeff_e
                           * prod_features[pterm.prod_features_key]
                           * prod_anlmtp_erased[pterm.prod_key];
                    sum_f += pterm.coeff_f
                           * prod_features[pterm.prod_features_key]
                           * prod_anlmtp_erased[pterm.prod_key];
                    /*
                    prod = prod_anlmtp_erased[pterm.prod_key]
                          * prod_features[pterm.prod_features_key];
                    sum_e += pterm.coeff_e * prod;
                    sum_f += pterm.coeff_f * prod;
                    */
                }
            }
            prod_sum_e[i][key] = sum_e;
            prod_sum_f[i][key] = sum_f;
        }

        clock_t t5 = clock();
   /*
        std::cout << "prod"
            << double(t2-t1)/CLOCKS_PER_SEC << " "
            << double(t3-t2)/CLOCKS_PER_SEC << " "
            << double(t4-t3)/CLOCKS_PER_SEC << " "
            << double(t5-t4)/CLOCKS_PER_SEC << " "
            << std::endl;
    */
    }
}

void PolymlpEval::compute_linear_features(const vector1d& prod_anlmtp,
                                          const int type1,
                                          vector1d& feature_values){

    const auto& linear_features = pot.p_obj.get_linear_features(type1);
    feature_values = vector1d(linear_features.size(), 0.0);

    int idx = 0;
    double val;
    for (const auto& sfeature: linear_features){
        val = 0.0;
        for (const auto& sterm: sfeature){
            val += sterm.coeff * prod_anlmtp[sterm.prod_key];
        }
        feature_values[idx] = val;
        ++idx;
    }
}
