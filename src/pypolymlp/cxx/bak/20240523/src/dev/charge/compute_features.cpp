/****************************************************************************

        Copyright (C) 2022 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute_features.h"

ComputeFeatures::ComputeFeatures(){}

ComputeFeatures::ComputeFeatures(const vector3d& dis_array,
                                 const vector4d& diff_array,
                                 const vector3i& atom2_array,
                                 const vector1i& types_i,
                                 const feature_params& fp){

    types = types_i;
    n_atom = dis_array.size();
    n_type = fp.n_type;

    bool icharge = true;
    modelp = ModelParams(fp, icharge);
    const Features f_obj(fp, modelp);
    n_terms = f_obj.get_n_feature_combinations();
    const vector1d pot(n_terms, 1.0);
    p_obj = Potential(f_obj, pot);

    if (icharge == true and fp.des_type == "pair"){
        vector2d antc;
        vector2map_d prod_sum_e;
        compute_antc(dis_array, fp, antc);
        compute_sum_of_prod_antc(antc, prod_sum_e);
        compute_features_charge(dis_array, atom2_array, fp, prod_sum_e);
    }
    else if (icharge == true and fp.des_type == "gtinv"){
        vector2dc anlmtc;
        vector2map_dc prod_sum_e;
        compute_anlmtc(dis_array, diff_array, fp, anlmtc);
        compute_sum_of_prod_anlmtc(anlmtc, prod_sum_e);
        compute_features_charge(dis_array,
                                diff_array,
                                atom2_array,
                                fp,
                                prod_sum_e);
    }
}

ComputeFeatures::~ComputeFeatures(){}

void ComputeFeatures::compute_features_charge(const vector3d& dis_array,
                                              const vector3i& atom2_array,
                                              const feature_params& fp,
                                              const vector2map_d& prod_sum_e){

    const auto& ntc_map = p_obj.get_ntc_map();

    vector1d fn;
    double val;

    xc = vector2d(n_atom, vector1d(n_terms, 0.0));
    const auto& type_comb_p = modelp.get_type_comb_pair();
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        const int type1 = types[atom1];
        for (int type2 = 0; type2 < n_type; ++type2){
            const auto& dis_a = dis_array[atom1][type2];
            const auto& atom2_a = atom2_array[atom1][type2];
            for (int j = 0; j < dis_a.size(); ++j){
                const double dis = dis_a[j];
                if (dis < fp.cutoff){
                    get_fn_(dis, fp, fn);
                    int head_key(0);
                    for (const auto& ntc: ntc_map){
                        if (type_comb_p[ntc.tc][type1].size() > 0
                            and type2 == type_comb_p[ntc.tc][type1][0]){
                            const int atom2 = atom2_a[j];
                            const auto& prod = prod_sum_e[atom1][head_key];
                            for (const auto& p_ele: prod){
                                val = fn[ntc.n] * p_ele.second;
                                xc[atom1][p_ele.first] += 0.5 * val;
                                xc[atom2][p_ele.first] -= 0.5 * val;
                            }
                        }
                        ++head_key;
                    }
                }
            }
        }
    }
}

void ComputeFeatures::compute_features_charge(const vector3d& dis_array,
                                              const vector4d& diff_array,
                                              const vector3i& atom2_array,
                                              const feature_params& fp,
                                              const vector2map_dc& prod_sum_e){

    const auto& nlmtc_map_no_conj = p_obj.get_nlmtc_map_no_conjugate();

    vector1d fn;
    vector1dc ylm;
    dc head_val;
    double val;

    xc = vector2d(n_atom, vector1d(n_terms, 0.0));
    const auto& type_comb_p = modelp.get_type_comb_pair();
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        const int type1 = types[atom1];
        for (int type2 = 0; type2 < n_type; ++type2){
            const auto& dis_a = dis_array[atom1][type2];
            const auto& diff_a = diff_array[atom1][type2];
            const auto& atom2_a = atom2_array[atom1][type2];
            for (int j = 0; j < dis_a.size(); ++j){
                const double dis = dis_a[j];
                if (dis < fp.cutoff){
                    const vector1d &sph = cartesian_to_spherical_(diff_a[j]);
                    get_fn_(dis, fp, fn);
                    get_ylm_(sph[0], sph[1], fp.maxl, ylm);
                    for (const auto& nlmtc: nlmtc_map_no_conj){
                        const int tc = nlmtc.tc;
                        if (type_comb_p[tc][type1].size() > 0
                            and type2 == type_comb_p[tc][type1][0]){
                            const int atom2 = atom2_a[j];
                            const int n = nlmtc.n;
                            const auto& lm_attr = nlmtc.lm;
                            const int ylm_key = lm_attr.ylmkey;
                            const int head_key = nlmtc.nlmtc_noconj_key;

                            head_val = fn[n] * ylm[ylm_key];
                            const auto& prod = prod_sum_e[atom1][head_key];
                            if (lm_attr.m == 0){
                                for (const auto& p_ele: prod){
                                    val = prod_real(head_val, p_ele.second);
                                    xc[atom1][p_ele.first] += 0.5 * val;
                                    xc[atom2][p_ele.first] -= 0.5 * val;
                                }
                            }
                            else {
                                for (const auto& p_ele: prod){
                                    val = prod_real(head_val, p_ele.second);
                                    xc[atom1][p_ele.first] += val;
                                    xc[atom2][p_ele.first] -= val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ComputeFeatures::compute_antc(const vector3d& dis_array,
                                   const feature_params& fp,
                                   vector2d& antc){

    const auto& ntc_map = p_obj.get_ntc_map();
    const auto& type_comb_p = modelp.get_type_comb_pair();
    const int inum = dis_array.size();

    antc = vector2d(inum, vector1d(ntc_map.size(), 0.0));

    vector1d fn;
    for (int i = 0; i < dis_array.size(); ++i) {
        int type1 = types[i];
        for (int type2 = 0; type2 < n_type; ++type2){
            const auto& dis_a = dis_array[i][type2];
            for (int j = 0; j < dis_a.size(); ++j){
                const double dis = dis_a[j];
                if (dis < fp.cutoff){
                    get_fn_(dis, fp, fn);
                    int idx(0);
                    for (const auto& ntc: ntc_map){
                        if (type_comb_p[ntc.tc][type1].size() > 0
                            and type2 == type_comb_p[ntc.tc][type1][0]){
                            antc[i][idx] += fn[ntc.n];
                        }
                        ++idx;
                    }
                }
            }
        }
    }
}

void ComputeFeatures::compute_sum_of_prod_antc(const vector2d& antc,
                                               vector2map_d& prod_sum_e){

    const auto& prod_map = p_obj.get_prod_map();
    const auto& prod_features_map = p_obj.get_prod_features_map();
    const auto& ntc_map = p_obj.get_ntc_map();
    const auto& linear_features = p_obj.get_linear_features();

    const int inum = antc.size();
    const int n_head_keys = ntc_map.size();

    prod_sum_e = vector2map_d(inum, vector1map_d(n_head_keys));
    for (int i = 0; i < inum; ++i) {

        // computing nonequivalent products of order parameters (antc)
        vector1d prod_antc;
        compute_products<double>(prod_map, antc[i], prod_antc);
        // end: computing products of order parameters (antc)

        // computing linear features
        vector1d features;
        for (const auto& sfeature: linear_features){
            features.emplace_back(prod_antc[sfeature[0].prod_key]);
        }
        // end: computing linear features

        vector1d prod_features;
        compute_products<double>(prod_features_map, features, prod_features);

        int key(0), f_idx;
        double prod, val;
        for (const auto& ntc: ntc_map){
            const auto& pmodel = p_obj.get_potential_model(ntc.ntc_key);
            std::map<int, double> psum;
            for (const auto& pterm: pmodel){
                prod = prod_antc[pterm.prod_key]
                     * prod_features[pterm.prod_features_key];
                val = pterm.coeff_e * prod;
                f_idx = pterm.feature_idx;
                if (psum.count(f_idx) == 0) psum[f_idx] = val;
                else psum[f_idx] += val;
            }
            prod_sum_e[i][key] = psum;
            ++key;
        }
    }
}

void ComputeFeatures::compute_anlmtc(const vector3d& dis_array,
                                     const vector4d& diff_array,
                                     const feature_params& fp,
                                     vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = p_obj.get_nlmtc_map_no_conjugate();
    const auto& type_comb_p = modelp.get_type_comb_pair();

    const int inum = dis_array.size();
    vector2d anlmtc_r(inum, vector1d(nlmtc_map_no_conj.size(), 0.0));
    vector2d anlmtc_i(inum, vector1d(nlmtc_map_no_conj.size(), 0.0));

    vector1d fn; vector1dc ylm; dc val;
    for (int i = 0; i < dis_array.size(); ++i) {
        int type1 = types[i];
        for (int type2 = 0; type2 < n_type; ++type2){
            const auto& dis_a = dis_array[i][type2];
            const auto& diff_a = diff_array[i][type2];
            for (int j = 0; j < dis_a.size(); ++j){
                const double dis = dis_a[j];
                if (dis < fp.cutoff){
                    const vector1d &sph = cartesian_to_spherical_(diff_a[j]);
                    get_fn_(dis, fp, fn);
                    get_ylm_(sph[0], sph[1], fp.maxl, ylm);
                    for (const auto& nlmtc: nlmtc_map_no_conj){
                        const int tc = nlmtc.tc;
                        if (type_comb_p[tc][type1].size() > 0
                            and type2 == type_comb_p[tc][type1][0]){
                            const auto& lm_attr = nlmtc.lm;
                            const int idx = nlmtc.nlmtc_noconj_key;
                            val = fn[nlmtc.n] * ylm[lm_attr.ylmkey];
                            anlmtc_r[i][idx] += val.real();
                            anlmtc_i[i][idx] += val.imag();
                        }
                    }
                }
            }
        }
    }
    compute_anlmtc_conjugate(anlmtc_r, anlmtc_i, anlmtc);
}

void ComputeFeatures::compute_anlmtc_conjugate(const vector2d& anlmtc_r,
                                               const vector2d& anlmtc_i,
                                               vector2dc& anlmtc){

    const auto& nlmtc_map_no_conj = p_obj.get_nlmtc_map_no_conjugate();
    const auto& n_nlmtc_all = p_obj.get_n_nlmtc_all();

    int inum = anlmtc_r.size();
    anlmtc = vector2dc(inum, vector1dc(n_nlmtc_all, 0.0));

    for (int i = 0; i < inum; ++i){
        for (const auto& nlmtc: nlmtc_map_no_conj){
            const int idx = nlmtc.nlmtc_noconj_key;
            const auto cc_coeff = nlmtc.lm.cc_coeff;
            anlmtc[i][nlmtc.nlmtc_key] = {anlmtc_r[i][idx],anlmtc_i[i][idx]};
            anlmtc[i][nlmtc.conj_key] = {cc_coeff * anlmtc_r[i][idx],
                                          - cc_coeff * anlmtc_i[i][idx]};
        }
    }
}

void ComputeFeatures::compute_sum_of_prod_anlmtc(const vector2dc& anlmtc,
                                                 vector2map_dc& prod_sum_e){

    const auto& prod_map = p_obj.get_prod_map();
    const auto& prod_map_erased = p_obj.get_prod_map_erased();
    const auto& prod_features_map = p_obj.get_prod_features_map();

    const auto& nlmtc_map_no_conj = p_obj.get_nlmtc_map_no_conjugate();
    const int n_head_keys = nlmtc_map_no_conj.size();
    const int inum = anlmtc.size();

    prod_sum_e = vector2map_dc(inum, vector1map_dc(n_head_keys));
    for (int i = 0; i < inum; ++i) {

        // computing nonequivalent products of order parameters (anlmtc)
        vector1dc prod_anlmtc, prod_anlmtc_erased;
        compute_products<dc>(prod_map, anlmtc[i], prod_anlmtc);
        compute_products<dc>(prod_map_erased, anlmtc[i], prod_anlmtc_erased);
        // end: computing products of order parameters (anlmtc)

        // computing linear features
        //   and nonequivalent products of linear features
        vector1d features, prod_features;
        compute_linear_features(prod_anlmtc, features);
        compute_products<double>(prod_features_map, features, prod_features);
        // end: computing linear features

        int idx;
        dc prod, val;
        for (int key = 0; key < nlmtc_map_no_conj.size(); ++key){
            std::map<int, dc> psum;
            const auto& pmodel = p_obj.get_potential_model(key);
            for (const auto& pterm: pmodel){
                prod = prod_anlmtc_erased[pterm.prod_key]
                     * prod_features[pterm.prod_features_key];
                val = pterm.coeff_e * prod;
                idx = pterm.feature_idx;
                if (psum.count(idx) == 0) psum[idx] = val;
                else psum[idx] += val;
            }
            prod_sum_e[i][key] = psum;
        }
    }
}

void ComputeFeatures::compute_linear_features(const vector1dc& prod_anlmtc,
                                              vector1d& feature_values){

    const auto& linear_features = p_obj.get_linear_features();
    feature_values.resize(linear_features.size());

    int idx = 0;
    for (const auto& sfeature: linear_features){
        dc val(0.0);
        for (const auto& sterm: sfeature){
            val += sterm.coeff * prod_anlmtc[sterm.prod_key];
        }
        feature_values[idx] = std::real(val);
        ++idx;
    }
}

template<typename T>
void ComputeFeatures::compute_products(const vector2i& map,
                                       const std::vector<T>& element,
                                       std::vector<T>& prod_vals){

    prod_vals = std::vector<T>(map.size());

    int idx(0);
    for (const auto& prod: map){
        T val;
        if (prod.size() == 0){
            val = 1.0;
        }
        else {
            val = element[prod[0]];
            for (int n = 1; n < prod.size(); ++n) val *= element[prod[n]];
        }
        prod_vals[idx] = val;
        ++idx;
    }
}

double ComputeFeatures::prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

const vector2d& ComputeFeatures::get_x() const {
    return xc;
}
