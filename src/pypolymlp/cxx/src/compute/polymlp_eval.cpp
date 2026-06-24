/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_eval.h"

PolymlpEval::PolymlpEval(){}

PolymlpEval::PolymlpEval(const feature_params& fp, const vector1d& coeffs){

    use_openmp = true;

    vector1d coeffs_rev(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) coeffs_rev[i] = 2.0 * coeffs[i];
    polymlp_api.set_potential_model(fp, coeffs_rev);

    const auto& maps = polymlp_api.get_maps();
    const auto& tp_to_params = maps.tp_to_params;
    int n_type = fp.n_type;
    int n_tp = tp_to_params.size();

    if (fp.feature_type == "pair"){
        /*** TODO: Pair mapping ***/
    }
    else if (fp.feature_type == "gtinv"){
        /*** TODO: Prepare in initialization ***/
        nlmtp_attrs.resize(n_type);
        for (int type1 = 0; type1 < n_type; ++type1){
            nlmtp_attrs[type1].resize(n_tp);
            const auto& maps_type = maps.maps_type[type1];
            const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;
            for (const auto& nlmtp: nlmtp_attrs_noconj){
                nlmtp_attrs[type1][nlmtp.tp].emplace_back(nlmtp);
            }
        }
    }
}

PolymlpEval::~PolymlpEval(){}

void PolymlpEval::eval(
    const vector1i& types,
    NeighborHalf& neigh,
    const bool use_openmp_,
    double& energy,
    vector2d& forces,
    vector1d& stress
){
    use_openmp = use_openmp_;
    n_atom = types.size();
    const auto& fp = polymlp_api.get_fp();

    if (fp.feature_type == "pair"){
        eval_pair(types, neigh, energy, forces, stress);
    }
    else if (fp.feature_type == "gtinv"){
        eval_gtinv(types, neigh, energy, forces, stress);
    }
}

/*--- feature_type = pair ----------------------------------------------*/

void PolymlpEval::eval_pair(
    const vector1i& types,
    NeighborHalf& neigh,
    double& energy,
    vector2d& forces,
    vector1d& stress
){
    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    vector2d antp, prod_sum_e, prod_sum_f;
    compute_antp(types, neigh, antp);
    compute_sum_of_prod_antp(types, antp, prod_sum_e, prod_sum_f);

    vector2d e_array(n_atom),fx_array(n_atom),fy_array(n_atom),fz_array(n_atom);
    for (int i = 0; i < n_atom; ++i) {
        int jsize = neigh.size(i);
        e_array[i].resize(jsize);
        fx_array[i].resize(jsize);
        fy_array[i].resize(jsize);
        fz_array[i].resize(jsize);
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (use_openmp)
    #endif
    for (int i = 0; i < n_atom; ++i) {
        int type1, type2, tp;
        double dx, dy, dz, dis, e_ij, f_ij, fx, fy, fz;
        vector1d fn, fn_d;

        type1 = types[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;

        auto [begin, end] = neigh.range(i);
        for (int k = begin; k < end; ++k) {
            int jj = k - begin;
            int j = neigh.neighbor_atom(k);
            neigh.diff_ij(k, dx, dy, dz);
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis >= fp.cutoff)
                continue;

            type2 = types[j];
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

            e_array[i][jj] = e_ij;
            fx_array[i][jj] = fx;
            fy_array[i][jj] = fy;
            fz_array[i][jj] = fz;
        }
    }
    collect_properties(
        e_array, fx_array, fy_array, fz_array, neigh,
        energy, forces, stress
    );
}

void PolymlpEval::compute_antp(
    const vector1i& types,
    NeighborHalf& neigh,
    vector2d& antp
){

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

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
        const auto& maps_type = maps.maps_type[type1];
        const auto& ntp_attrs = maps_type.ntp_attrs;

        auto [begin, end] = neigh.range(i);
        for (int k = begin; k < end; ++k) {
            int jj = k - begin;
            int j = neigh.neighbor_atom(k);
            neigh.diff_ij(k, dx, dy, dz);
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis >= fp.cutoff)
                continue;

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


void PolymlpEval::compute_sum_of_prod_antp(
    const vector1i& types,
    const vector2d& antp,
    vector2d& prod_sum_e,
    vector2d& prod_sum_f){


    prod_sum_e = vector2d(n_atom);
    prod_sum_f = vector2d(n_atom);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (use_openmp)
    #endif
    for (int i = 0; i < n_atom; ++i) {
        const int type1 = types[i];
        polymlp_api.compute_sum_of_prod_antp(
            antp[i], type1, prod_sum_e[i], prod_sum_f[i]
        );
    }
}

/*******************************************************************************
  Feature_type = gtinv
********************************************************************************/

void PolymlpEval::eval_gtinv(
    const vector1i& types,
    NeighborHalf& neigh,
    double& energy,
    vector2d& forces,
    vector1d& stress){

    //auto start1 = std::chrono::high_resolution_clock::now();
    //vector2dc anlmtp, prod_sum_e, prod_sum_f;
    //auto end1 = std::chrono::high_resolution_clock::now();
    //auto elapsed1 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    //std::cout << "Elapsed time: " << elapsed1.count() << " micro s" << std::endl;

    //auto start2 = std::chrono::high_resolution_clock::now();
    vector2dc prod_sum_e, prod_sum_f;
    compute_sum_of_prod_anlmtp(types, neigh, prod_sum_e, prod_sum_f);
    //auto end2 = std::chrono::high_resolution_clock::now();
    //auto elapsed2 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    //std::cout << "Elapsed time: " << elapsed2.count() << " micro s" << std::endl;

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    vector2d e_array(n_atom),fx_array(n_atom),fy_array(n_atom),fz_array(n_atom);
    for (int i = 0; i < n_atom; ++i) {
        int jsize = neigh.size(i);
        e_array[i].resize(jsize);
        fx_array[i].resize(jsize);
        fy_array[i].resize(jsize);
        fz_array[i].resize(jsize);
    }

    auto start3 = std::chrono::high_resolution_clock::now();
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (use_openmp)
    #endif
    for (int i = 0; i < n_atom; ++i) {
        int type1, type2, tp;
        double dx, dy, dz, dis;
        double e_ij, fx, fy, fz;
        dc val, valx, valy, valz, d1;
        vector1d fn, fn_d;
        vector1dc ylm, ylm_dx, ylm_dy, ylm_dz;

        type1 = types[i];
        const auto& nlmtp_attrs1 = nlmtp_attrs[type1];

        auto [begin, end] = neigh.range(i);
        for (int k = begin; k < end; ++k) {
            int jj = k - begin;
            int j = neigh.neighbor_atom(k);
            // (dx, dy, dz) = pos[i] - pos[j]
            neigh.diff_ij(k, dx, dy, dz);
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis >= fp.cutoff)
                continue;

            type2 = types[j];
            tp = type_pairs[type1][type2];
            const auto& params = tp_to_params[tp];
            get_fn_(dis, fp, params, fn, fn_d);
            get_ylm_(dis, dx, dy, dz, fp.maxl, ylm, ylm_dx, ylm_dy, ylm_dz);

            e_ij = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
            const auto& attrs = nlmtp_attrs1[tp];

            for (const auto& nlmtp : attrs){
                const int nid = nlmtp.n_id;
                double fn_val = fn[nid];
                if (fn_val < 1e-20)
                    continue;

                const auto& lm_attr = nlmtp.lm;
                const int ylmkey = lm_attr.ylmkey;
                const int idx_i = nlmtp.ilocal_noconj_id;
                const int idx_j = nlmtp.jlocal_noconj_id;
                val = fn_val * ylm[ylmkey];
                d1 = fn_d[nid] * ylm[ylmkey] / dis;
                valx = - (d1 * dx + fn_val * ylm_dx[ylmkey]);
                valy = - (d1 * dy + fn_val * ylm_dy[ylmkey]);
                valz = - (d1 * dz + fn_val * ylm_dz[ylmkey]);

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
            e_array[i][jj] = e_ij;
            fx_array[i][jj] = fx;
            fy_array[i][jj] = fy;
            fz_array[i][jj] = fz;
        }
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    //auto elapsed3 =
    //    std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);
    //std::cout << "Elapsed time: " << elapsed3.count() << " micro s" << std::endl;

    collect_properties(
        e_array, fx_array, fy_array, fz_array, neigh,
        energy, forces, stress
    );
}


void PolymlpEval::collect_properties(
    const vector2d& e_array,
    const vector2d& fx_array,
    const vector2d& fy_array,
    const vector2d& fz_array,
    NeighborHalf& neigh,
    double& energy,
    vector2d& forces,
    vector1d& stress
){
    const auto& fp = polymlp_api.get_fp();

    energy = 0.0;
    forces.resize(n_atom);
    for (int i = 0; i < n_atom; ++i) forces[i] = vector1d(3, 0.0);
    stress = vector1d(6, 0.0);

    double dx, dy, dz, dis, fx, fy, fz;
    for (int i = 0; i < n_atom; ++i) {
        auto [begin, end] = neigh.range(i);
        for (int k = begin; k < end; ++k) {
            int jj = k - begin;
            int j = neigh.neighbor_atom(k);
            neigh.diff_ij(k, dx, dy, dz);
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis >= fp.cutoff)
                continue;

            energy += e_array[i][jj];
            fx = fx_array[i][jj];
            fy = fy_array[i][jj];
            fz = fz_array[i][jj];
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

void PolymlpEval::compute_sum_of_prod_anlmtp(
    const vector1i& types,
    NeighborHalf& neigh,
    vector2dc& prod_sum_e,
    vector2dc& prod_sum_f)
{
    prod_sum_e = vector2dc(n_atom);
    prod_sum_f = vector2dc(n_atom);

    const auto& fp = polymlp_api.get_fp();
    const auto& maps = polymlp_api.get_maps();
    const auto& type_pairs = maps.type_pairs;
    const auto& tp_to_params = maps.tp_to_params;

    vector1i offset_full, neighbor_full;
    vector1d dx_full, dy_full, dz_full;
    neigh.get_full_list(neighbor_full, dx_full, dy_full, dz_full, offset_full);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (use_openmp)
    #endif
    for (int i = 0; i < n_atom; ++i) {
        int type1, type2, tp;
        double dx, dy, dz, dis;
        vector1d fn; vector1dc ylm; dc val;

        type1 = types[i];
        const auto& maps_type = maps.maps_type[type1];
        const auto& nlmtp_attrs_noconj = maps_type.nlmtp_attrs_noconj;
        const auto& nlmtp_attrs1 = nlmtp_attrs[type1];

        vector1d anlmtp_r(nlmtp_attrs_noconj.size(), 0.0);
        vector1d anlmtp_i(nlmtp_attrs_noconj.size(), 0.0);
        for (int k = offset_full[i]; k < offset_full[i + 1]; ++k){
            int j = neighbor_full[k];
            dx = - dx_full[k];
            dy = - dy_full[k];
            dz = - dz_full[k];
            dis = sqrt(dx*dx + dy*dy + dz*dz);
            if (dis >= fp.cutoff)
                continue;

            type2 = types[j];
            tp = type_pairs[type1][type2];
            const auto& params = tp_to_params[tp];
            get_fn_(dis, fp, params, fn);
            get_ylm_(dx, dy, dz, fp.maxl, ylm);

            const auto& attrs = nlmtp_attrs1[tp];
            for (const auto& nlmtp : attrs){
                double val_fn = fn[nlmtp.n_id];
                if (val_fn < 1e-20)
                    continue;

                const auto& lm_attr = nlmtp.lm;
                const int idx_i = nlmtp.ilocal_noconj_id;
                dc& val_ylm = ylm[lm_attr.ylmkey];
                val = val_fn * val_ylm;
                anlmtp_r[idx_i] += val.real();
                anlmtp_i[idx_i] += val.imag();
            }
        }
        vector1dc anlmtp;
        polymlp_api.compute_anlmtp_conjugate(anlmtp_r, anlmtp_i, types[i], anlmtp);
        polymlp_api.compute_sum_of_prod_anlmtp(
            anlmtp, types[i], prod_sum_e[i], prod_sum_f[i]
        );
    }
}
