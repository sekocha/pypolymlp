/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_eval.h"


PolymlpEval::PolymlpEval(){}

PolymlpEval::PolymlpEval(const feature_params& fp, const vector1d& coeffs){

    vector1d coeffs_rev(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) coeffs_rev[i] = 2.0 * coeffs[i];

    const bool icharge = false;
    pot.fp = fp;
    pot.modelp = ModelParams(pot.fp, icharge);
    const Features f_obj(pot.fp, pot.modelp);
    pot.p_obj = Potential(f_obj, coeffs_rev);
    set_type_comb();

}

PolymlpEval::~PolymlpEval(){}

void PolymlpEval::set_type_comb(){

    type_comb = vector2i(pot.fp.n_type, vector1i(pot.fp.n_type));
    for (int type1 = 0; type1 < pot.fp.n_type; ++type1){
        for (int type2 = 0; type2 < pot.fp.n_type; ++type2){
            for (size_t i = 0; i < pot.modelp.get_type_comb_pair().size(); ++i){
                const auto &tc = pot.modelp.get_type_comb_pair()[i];
                if (tc[type1].size() > 0 and tc[type1][0] == type2){
                    type_comb[type1][type2] = i;
                    break;
                }
            }
        }
    }
}

void PolymlpEval::eval(const vector2d& positions_c,
                       const vector1i& types,
                       const vector2i& neighbor_half,
                       const vector3d& neighbor_diff,
                       double& energy,
                       vector2d& forces,
                       vector1d& stress){

    if (pot.fp.des_type == "pair"){
        eval_pair(positions_c, types, neighbor_half, neighbor_diff,
                  energy, forces, stress);
    }
    else if (pot.fp.des_type == "gtinv"){
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

    const auto& ntc_map = pot.p_obj.get_ntc_map();

    const int n_atom = types.size();
    vector2d antc, prod_sum_e, prod_sum_f;
    compute_antc(positions_c, types, neighbor_half, neighbor_diff, antc);
    compute_sum_of_prod_antc(types, antc, prod_sum_e, prod_sum_f);

    energy = 0.0;
    forces.resize(n_atom);
    for (int i = 0; i < n_atom; ++i) forces[i] = vector1d(3, 0.0);
    stress = vector1d(6, 0.0);

    int type1,type2;
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
                get_fn_(dis, pot.fp, fn, fn_d);
                e_ij = 0.0, f_ij = 0.0;
                int head_key(0);
                for (const auto& ntc: ntc_map){
                    if (type_comb[type1][type2] == ntc.tc){
                        const auto& prod_ei = prod_sum_e[i][head_key];
                        const auto& prod_ej = prod_sum_e[j][head_key];
                        const auto& prod_fi = prod_sum_f[i][head_key];
                        const auto& prod_fj = prod_sum_f[j][head_key];
                        e_ij += fn[ntc.n] * (prod_ei + prod_ej);
                        f_ij += fn_d[ntc.n] * (prod_fi + prod_fj);
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

void PolymlpEval::compute_antc(const vector2d& positions_c,
                               const vector1i& types,
                               const vector2i& neighbor_half,
                               const vector3d& neighbor_diff,
                               vector2d& antc){

    const int n_atom = types.size();
    const auto& ntc_map = pot.p_obj.get_ntc_map();

    antc = vector2d(n_atom, vector1d(ntc_map.size(), 0.0));

    int type1, type2;
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
                get_fn_(dis, pot.fp, fn);
                int idx(0);
                for (const auto& ntc: ntc_map){
                    if (type_comb[type1][type2] == ntc.tc){
                        antc[i][idx] += fn[ntc.n];
                        antc[j][idx] += fn[ntc.n];
                    }
                    ++idx;
                }
            }
        }
    }
}

void PolymlpEval::compute_sum_of_prod_antc(const vector1i& types,
                                           const vector2d& antc,
                                           vector2d& prod_antc_sum_e,
                                           vector2d& prod_antc_sum_f){

    const auto& ntc_map = pot.p_obj.get_ntc_map();

    const int n_atom = antc.size();
    prod_antc_sum_e = vector2d(n_atom, vector1d(ntc_map.size(), 0.0));
    prod_antc_sum_f = vector2d(n_atom, vector1d(ntc_map.size(), 0.0));

    for (int i = 0; i < n_atom; ++i) {
        int type1 = types[i];
        const auto& linear_features = pot.p_obj.get_linear_features(type1);
        const auto& prod_map = pot.p_obj.get_prod_map(type1);
        const auto& prod_features_map = pot.p_obj.get_prod_features_map(type1);

        // computing products of order parameters (antc)
        vector1d prod_antc;
        compute_products<double>(prod_map, antc[i], prod_antc);
        // end: computing products of order parameters (antc)

        // computing linear features
        vector1d feature_values(linear_features.size(), 0.0);
        int idx = 0;
        for (const auto& sfeature: linear_features){
            if (sfeature.size() > 0){
                feature_values[idx] = prod_antc[sfeature[0].prod_key];
            }
            ++idx;
        }
        // end: computing linear features

        vector1d prod_features;
        compute_products<double>(prod_features_map,
                                 feature_values,
                                 prod_features);

        idx = 0;
        for (const auto& ntc: ntc_map){
            const auto& pmodel = pot.p_obj.get_potential_model(type1,
                                                               ntc.ntc_key);
            double sum_e(0.0), sum_f(0.0), prod;
            for (const auto& pterm: pmodel){
                prod = prod_antc[pterm.prod_key]
                     * prod_features[pterm.prod_features_key];
                sum_e += pterm.coeff_e * prod;
                sum_f += pterm.coeff_f * prod;
            }
            prod_antc_sum_e[i][idx] = 0.5 * sum_e;
            prod_antc_sum_f[i][idx] = 0.5 * sum_f;
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

    vector2dc anlmtc, prod_sum_e, prod_sum_f;
    clock_t t1 = clock();
    compute_anlmtc(positions_c, types, neighbor_half, neighbor_diff, anlmtc);
    clock_t t2 = clock();
    compute_sum_of_prod_anlmtc(types, anlmtc, prod_sum_e, prod_sum_f);
    clock_t t3 = clock();

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();

    energy = 0.0;
    forces.resize(n_atom);
    for (int i = 0; i < n_atom; ++i) forces[i] = vector1d(3, 0.0);
    stress = vector1d(6, 0.0);

    int type1,type2;
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
                const auto& sph= cartesian_to_spherical_(vector1d{dx,dy,dz});
                get_fn_(dis, pot.fp, fn, fn_d);
                get_ylm_(dis, sph[0], sph[1], pot.fp.maxl,
                         ylm, ylm_dx, ylm_dy, ylm_dz);

                e_ij = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
                const int tc12 = type_comb[type1][type2];
                for (const auto& nlmtc: nlmtc_map_no_conj){
                    const auto& lm_attr = nlmtc.lm;
                    const int ylmkey = lm_attr.ylmkey;
                    const int head_key = nlmtc.nlmtc_noconj_key;
                    if (tc12 == nlmtc.tc){
                        val = fn[nlmtc.n] * ylm[ylmkey];
                        d1 = fn_d[nlmtc.n] * ylm[ylmkey] / dis;
                        valx = - (d1 * dx + fn[nlmtc.n] * ylm_dx[ylmkey]);
                        valy = - (d1 * dy + fn[nlmtc.n] * ylm_dy[ylmkey]);
                        valz = - (d1 * dz + fn[nlmtc.n] * ylm_dz[ylmkey]);
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


void PolymlpEval::compute_anlmtc(const vector2d& positions_c,
                                 const vector1i& types,
                                 const vector2i& neighbor_half,
                                 const vector3d& neighbor_diff,
                                 vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();

    const int n_atom = types.size();
    vector2d anlmtc_r(n_atom, vector1d(nlmtc_map_no_conj.size(), 0.0));
    vector2d anlmtc_i(n_atom, vector1d(nlmtc_map_no_conj.size(), 0.0));

    int type1, type2;
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
                get_fn_(dis, pot.fp, fn);
                get_ylm_(sph[0], sph[1], pot.fp.maxl, ylm);
                const int tc12 = type_comb[type1][type2];
                for (const auto& nlmtc: nlmtc_map_no_conj){
                    const auto& lm_attr = nlmtc.lm;
                    const int idx = nlmtc.nlmtc_noconj_key;
                    if (tc12 == nlmtc.tc){
                        val = fn[nlmtc.n] * ylm[lm_attr.ylmkey];
                        anlmtc_r[i][idx] += val.real();
                        anlmtc_r[j][idx] += val.real() * lm_attr.sign_j;
                        anlmtc_i[i][idx] += val.imag();
                        anlmtc_i[j][idx] += val.imag() * lm_attr.sign_j;
                    }
                }
            }
        }
    }
    compute_anlmtc_conjugate(anlmtc_r, anlmtc_i, anlmtc);
}


void PolymlpEval::compute_anlmtc_conjugate(const vector2d& anlmtc_r,
                                           const vector2d& anlmtc_i,
                                           vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();
    const auto& n_nlmtc_all = pot.p_obj.get_n_nlmtc_all();
    const int n_atom = anlmtc_r.size();
    anlmtc = vector2dc(n_atom, vector1dc(n_nlmtc_all, 0.0));

//    #ifdef _OPENMP
//    #pragma omp parallel for schedule(guided)
//    #endif
    for (int i = 0; i < n_atom; ++i) {
        int idx(0);
        for (const auto& nlmtc: nlmtc_map_no_conj){
            const auto& cc_coeff = nlmtc.lm.cc_coeff;
            anlmtc[i][nlmtc.nlmtc_key] = {anlmtc_r[i][idx], anlmtc_i[i][idx]};
            anlmtc[i][nlmtc.conj_key] = {cc_coeff * anlmtc_r[i][idx],
                                          - cc_coeff * anlmtc_i[i][idx]};
            ++idx;
        }
    }
}


void PolymlpEval::compute_sum_of_prod_anlmtc(const vector1i& types,
                                             const vector2dc& anlmtc,
                                             vector2dc& prod_sum_e,
                                             vector2dc& prod_sum_f){

    const auto& nlmtc_map_no_conj = pot.p_obj.get_nlmtc_map_no_conjugate();
    const int n_head_keys = nlmtc_map_no_conj.size();
    const int n_atom = types.size();
    prod_sum_e = vector2dc(n_atom, vector1dc(n_head_keys));
    prod_sum_f = vector2dc(n_atom, vector1dc(n_head_keys));

    for (int i = 0; i < n_atom; ++i) {
        int type1 = types[i];

        const auto& prod_map = pot.p_obj.get_prod_map(type1);
        const auto& prod_map_erased = pot.p_obj.get_prod_map_erased(type1);
        const auto& prod_features_map = pot.p_obj.get_prod_features_map(type1);

        clock_t t1 = clock();
        // computing nonequivalent products of order parameters (anlmtc)
        vector1d prod_anlmtc;
        compute_products_real(prod_map, anlmtc[i], prod_anlmtc);
        clock_t t2 = clock();
        // end: computing products of order parameters (anlmtc)

        // computing linear features
        //   and nonequivalent products of linear features
        vector1d features, prod_features;
        compute_linear_features(prod_anlmtc, type1, features);
        compute_products<double>(prod_features_map, features, prod_features);
        // end: computing linear features
        clock_t t3 = clock();

        vector1dc prod_anlmtc_erased;
        compute_products<dc>(prod_map_erased, anlmtc[i], prod_anlmtc_erased);
        clock_t t4 = clock();

        for (size_t key = 0; key < nlmtc_map_no_conj.size(); ++key){
            const auto& pmodel = pot.p_obj.get_potential_model(type1, key);
            dc sum_e(0.0), sum_f(0.0);
            //dc prod;
            for (const auto& pterm: pmodel){
                // TODO: examine accuracy
                if (fabs(prod_features[pterm.prod_features_key]) > 1e-50){
                    sum_e += pterm.coeff_e
                           * prod_features[pterm.prod_features_key]
                           * prod_anlmtc_erased[pterm.prod_key];
                    sum_f += pterm.coeff_f
                           * prod_features[pterm.prod_features_key]
                           * prod_anlmtc_erased[pterm.prod_key];
                    /*
                    prod = prod_anlmtc_erased[pterm.prod_key]
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

void PolymlpEval::compute_linear_features(const vector1d& prod_anlmtc,
                                          const int type1,
                                          vector1d& feature_values){

    const auto& linear_features = pot.p_obj.get_linear_features(type1);
    feature_values = vector1d(linear_features.size(), 0.0);

    int idx = 0;
    double val;
    for (const auto& sfeature: linear_features){
        val = 0.0;
        for (const auto& sterm: sfeature){
            val += sterm.coeff * prod_anlmtc[sterm.prod_key];
        }
        feature_values[idx] = val;
        ++idx;
    }
}

template<typename T>
void PolymlpEval::compute_products(const vector2i& map,
                                   const std::vector<T>& element,
                                   std::vector<T>& prod_vals){

    prod_vals = std::vector<T>(map.size());

    int idx(0);
    T val_p;
    for (const auto& prod: map){
        if (prod.size() > 0){
            auto iter = prod.begin();
            val_p = element[*iter];
            ++iter;
            while (iter != prod.end()){
                val_p *= element[*iter];
                ++iter;
            }
        }
        else val_p = 1.0;

        prod_vals[idx] = val_p;
        ++idx;
    }
}

void PolymlpEval::compute_products_real(const vector2i& map,
                                        const vector1dc& element,
                                        vector1d& prod_vals){

    prod_vals = vector1d(map.size());

    int idx(0);
    dc val_p;
    for (const auto& prod: map){
        if (prod.size() > 1) {
            auto iter = prod.begin() + 1;
            val_p = element[*iter];
            ++iter;
            while (iter != prod.end()){
                val_p *= element[*iter];
                ++iter;
            }
            prod_vals[idx] = prod_real(val_p, element[*(prod.begin())]);
        }
        else if (prod.size() == 1){
            prod_vals[idx] = element[*(prod.begin())].real();
        }
        else prod_vals[idx] = 1.0;
        ++idx;
    }
}

double PolymlpEval::prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

dc PolymlpEval::prod_real_and_complex(const double val1, const dc& val2){
    return dc(val1 * val2.real(), val1 * val2.imag());
}
