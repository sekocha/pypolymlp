/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_parse_potential_legacy.h"

bool parse_elements(std::ifstream& input, std::vector<std::string>& ele){

    std::stringstream ss;
    std::string line, tmp;

    ele.clear();
    std::getline( input, line );

    bool legacy;
    if (line.find("# element") != std::string::npos) legacy = true;
    else legacy = false;

    ss << line;
    while (!ss.eof()){
        ss >> tmp;
        ele.push_back(tmp);
    }
    ele.erase(ele.end()-1);
    ele.erase(ele.end()-1);
    ss.str("");
    ss.clear(std::stringstream::goodbit);

    return legacy;
}

void parse_basic_params(std::ifstream& input, feature_params& fp){

    fp.cutoff = get_value<double>(input);
    fp.pair_type = get_value<std::string>(input);
    fp.feature_type = get_value<std::string>(input);
    fp.model_type = get_value<int>(input);
    fp.maxp = get_value<int>(input);
    fp.maxl = get_value<int>(input);
}

int parse_gtinv_params(
    std::ifstream& input,
    vector1i& gtinv_maxl,
    std::vector<bool>& gtinv_sym
){
    int gtinv_order = get_value<int>(input);
    int size = gtinv_order - 1;
    gtinv_maxl = get_value_array<int>(input, size);
    gtinv_sym = get_value_array<bool>(input, size);
    return gtinv_order;
}

void parse_reg_coeffs(std::ifstream& input, vector1d& reg_coeffs){

    int n_reg_coeffs = get_value<int>(input);
    reg_coeffs = get_value_array<double>(input, n_reg_coeffs);
    vector1d scale = get_value_array<double>(input, n_reg_coeffs);
    for (int i = 0; i < n_reg_coeffs; ++i) reg_coeffs[i] *= 2.0/scale[i];
}

int parse_gaussians(std::ifstream& input, feature_params& fp){

    int n_params = get_value<int>(input);
    fp.params = vector2d(n_params);
    for (int i = 0; i < n_params; ++i)
        fp.params[i] = get_value_array<double>(input, 2);
    return n_params;
}

void parse_params_conditional(std::ifstream& input, feature_params& fp){

    fp.params_conditional = vector3i(fp.n_type, vector2i(fp.n_type));
    int n_type_pairs = get_value<int>(input);
    if (!input.eof() and n_type_pairs > 0){
        for (int i = 0; i < n_type_pairs; ++i){
            vector1i atomtypes = get_value_array<int>(input, 3);
            const int n_active = atomtypes[2];
            vector1i param_indices = get_value_array<int>(input, n_active);
            fp.params_conditional[atomtypes[0]][atomtypes[1]] = param_indices;
        }
    }
    else {
        const int n_params = fp.params.size();
        vector1i param_indices;
        for (int i = 0; i < n_params; ++i) param_indices.emplace_back(i);
        for (int i = 0; i < fp.n_type; ++i){
            for (int j = 0; j <= i; ++j){
                fp.params_conditional[j][i] = param_indices;
            }
        }
    }
}

bool parse_type_indices(
    std::ifstream& input, feature_params& fp, vector1i& type_indices
){

    bool n_type_full = get_value<bool>(input);
    if (!input.eof()){
        type_indices = get_value_array<int>(input, fp.n_type);
    }
    else {
        n_type_full = true;
        for (int i = 0; i < fp.n_type; ++i) type_indices.emplace_back(i);
    }
    return n_type_full;
}

bool check_polymlp_legacy(const char *file){
    std::ifstream input(file);
    if (input.fail()){
        std::cerr << "Error: Could not open mlp file: " << file << "\n";
        exit(8);
    }
    std::vector<std::string> ele;
    bool legacy = parse_elements(input, ele);
    return legacy;
}

void parse_polymlp_legacy(
    const char *file,
    feature_params& fp,
    vector1d& reg_coeffs,
    std::vector<std::string>& ele,
    vector1d& mass
){

    std::ifstream input(file);
    if (input.fail()){
        std::cerr << "Error: Could not open mlp file: " << file << "\n";
        exit(8);
    }

    fp.force = true;

    // line 1: elements
    parse_elements(input, ele);
    fp.n_type = int(ele.size());

    // line 2-7: cutoff radius, pair type, descriptor type, model_type, max power, max l
    parse_basic_params(input, fp);

    // line 8-10: gtinv_order, gtinv_maxl and gtinv_sym (optional)
    int gtinv_order;
    vector1i gtinv_maxl;
    std::vector<bool> gtinv_sym;
    if (fp.feature_type == "gtinv"){
        gtinv_order = parse_gtinv_params(input, gtinv_maxl, gtinv_sym);
    }

    // line 11: number of regression coefficients
    // line 12,13: regression coefficients, scale coefficients
    parse_reg_coeffs(input, reg_coeffs);

    // line 14: number of gaussian parameters
    // line 15-: gaussian parameters
    const int n_params = parse_gaussians(input, fp);

    // atomic mass, electrostatic
    mass = get_value_array<double>(input, ele.size());
    const bool icharge = get_value<bool>(input);

    // gtinv version and set gtinv attributes (optional)
    if (fp.feature_type == "gtinv"){
        int version = get_value<int>(input);
        if (version != 2) version = 1;

        // version must be implemented.
        Readgtinv rgt(gtinv_order, gtinv_maxl, gtinv_sym, ele.size(), version);
        fp.lm_array = rgt.get_lm_seq();
        fp.l_comb = rgt.get_l_comb();
        fp.lm_coeffs = rgt.get_lm_coeffs();
    }

    // params_conditional (optional)
    parse_params_conditional(input, fp);

    // type_indices (optional)
    vector1i type_indices;
    bool n_type_full = parse_type_indices(input, fp, type_indices);

}
