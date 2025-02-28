/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "polymlp_parse_potential.h"

std::string replace(std::string target, std::string str1, std::string str2){

    std::string::size_type  pos(target.find(str1));
    while(pos != std::string::npos){
        target.replace(pos, str1.length(), str2);
        pos = target.find(str1, pos + str2.length());
    }
    return target;
}

vector1str split(const std::string& s, char delim){

    vector1str elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

vector1str split_list(const std::string& s){

    vector1str elems;
    std::stringstream ss;
    std::string tmp;

    std::string::size_type pos = s.find("[");
    if (pos != std::string::npos){
        std::string str = split(s, '[')[1];
        str = split(str, ']')[0];
        str = replace(str, ",", " ");
        ss << str;
        while (!ss.eof()){
            ss >> tmp;
            elems.push_back(tmp);
        }
    }
    else {
        ss << s;
        ss >> tmp;
        elems.push_back(tmp);
    }
    return elems;
}

ParsePolymlpYaml::ParsePolymlpYaml(){}

ParsePolymlpYaml::ParsePolymlpYaml(const char *file){

    std::ifstream ifs(file);
    if (ifs.fail()){
        std::cerr << "Error: Could not open mlp file: " << file << "\n";
        exit(8);
    }

    std::string line, key;
    while (getline(ifs, line)){
        const auto sp = split(line, ':');
        if (sp.size() > 0){
            std::stringstream ss;
            ss << sp[0];
            ss >> key;
            if (sp.size() > 1) {
                params[key] = split_list(sp[1]);
                assign_exceptional_parameters(ifs, key);
            }
        }
    }
}

ParsePolymlpYaml::~ParsePolymlpYaml(){}

void ParsePolymlpYaml::assign_exceptional_parameters(
    std::ifstream& ifs, const std::string& key
){

    int n_lines;
    std::string line;
    if (key == "n_pair_params"){
        n_lines = std::stoi(params[key][0]);
        pair_params = parse_2d(ifs, n_lines);
    }
    else if (key == "n_type_pairs"){
        int n_types = params["elements"].size();
        pair_params_conditional.resize(n_types);
        for (int i = 0; i < n_types; ++i){
            pair_params_conditional[i].resize(n_types);
        }

        n_lines = std::stoi(params[key][0]);
        getline(ifs, line);
        vector1str strs1, strs2;
        for (int i = 0; i < n_lines; ++i){
            getline(ifs, line);
            strs1 = split_list(line);
            getline(ifs, line);
            strs2 = split_list(line);
            int i1 = std::stoi(strs1[0]);
            int i2 = std::stoi(strs1[1]);
            pair_params_conditional[i1][i2] = strs2;
        }
    }
}

vector2str ParsePolymlpYaml::parse_2d(std::ifstream& ifs, const int n_lines){

    vector2str array2d;
    std::string line;
    getline(ifs, line);
    for (int i = 0; i < n_lines; ++i){
        getline(ifs, line);
        array2d.emplace_back(split_list(line));
    }
    return array2d;
}

vector1i ParsePolymlpYaml::transform_vector1i(vector1str& strings){
    vector1i ints;
    std::transform(strings.begin(), strings.end(), std::back_inserter(ints),
        [&](std::string s) {
            return std::stoi(s);
        });
    return ints;
}

vector1d ParsePolymlpYaml::transform_vector1d(vector1str& strings){
    vector1d doubles;
    std::transform(strings.begin(), strings.end(), std::back_inserter(doubles),
        [&](std::string s) {
            return std::stod(s);
        });
    return doubles;
}

const vector1str& ParsePolymlpYaml::get_elements(){
    return params["elements"];
}
const double ParsePolymlpYaml::get_cutoff(){
    return std::stod(params["cutoff"][0]);
}
const std::string& ParsePolymlpYaml::get_pair_type(){
    return params["pair_type"][0];
}
const std::string& ParsePolymlpYaml::get_feature_type(){
    return params["feature_type"][0];
}
const int ParsePolymlpYaml::get_max_p(){
    return std::stoi(params["max_p"][0]);
}
const int ParsePolymlpYaml::get_max_l(){
    return std::stoi(params["max_l"][0]);
}
const int ParsePolymlpYaml::get_model_type(){
    return std::stoi(params["model_type"][0]);
}

const int ParsePolymlpYaml::get_gtinv_order(){
    return std::stoi(params["gtinv_order"][0]);
}
const int ParsePolymlpYaml::get_gtinv_version(){
    return std::stoi(params["gtinv_version"][0]);
}
const vector1i ParsePolymlpYaml::get_gtinv_maxl(){
    return transform_vector1i(params["gtinv_maxl"]);
}
const vector1b ParsePolymlpYaml::get_gtinv_sym(){
    auto sym = transform_vector1i(params["gtinv_sym"]);
    vector1b sym_b;
    for (auto& s: sym) sym_b.emplace_back(static_cast<bool>(s));
    return sym_b;
}
const vector1d ParsePolymlpYaml::get_mass(){
    return transform_vector1d(params["mass"]);
}
const vector1d ParsePolymlpYaml::get_coeffs(){
    return transform_vector1d(params["coeffs"]);
}

const vector2d ParsePolymlpYaml::get_pair_params(){

    vector2d pair_params_double;
    for (auto& str1: pair_params){
        pair_params_double.emplace_back(transform_vector1d(str1));
    }
    return pair_params_double;
}
const vector3i ParsePolymlpYaml::get_pair_params_conditional(){

    vector3i pair_params_conditional_int(pair_params_conditional.size());
    int i = 0;
    for (auto& str1: pair_params_conditional){
        for (auto& str2: str1){
            pair_params_conditional_int[i].emplace_back(transform_vector1i(str2));
        }
        ++i;
    }
    return pair_params_conditional_int;
}

const int ParsePolymlpYaml::get_type_full(){
    return std::stoi(params["type_full"][0]);
}
const vector1i ParsePolymlpYaml::get_type_indices(){
    return transform_vector1i(params["type_indices"]);
}


void parse_polymlp(
    const char *file,
    feature_params& fp,
    vector1d& reg_coeffs,
    std::vector<std::string>& ele,
    vector1d& mass
){

    fp.force = true;
    auto yaml = ParsePolymlpYaml("polymlp.yaml");

    ele = yaml.get_elements();
    fp.n_type = int(ele.size());
    fp.cutoff = yaml.get_cutoff();
    fp.pair_type = yaml.get_pair_type();
    fp.feature_type = yaml.get_feature_type();
    fp.model_type = yaml.get_model_type();
    fp.maxp = yaml.get_max_p();
    fp.maxl = yaml.get_max_l();

    if (fp.feature_type == "gtinv"){
        int gtinv_order = yaml.get_gtinv_order();
        auto gtinv_maxl = yaml.get_gtinv_maxl();
        std::vector<bool> gtinv_sym = yaml.get_gtinv_sym();
        int version = yaml.get_gtinv_version();
        if (version != 2) version = 1;

        // version must be implemented.
        Readgtinv rgt(gtinv_order, gtinv_maxl, gtinv_sym, ele.size(), version);
        fp.lm_array = rgt.get_lm_seq();
        fp.l_comb = rgt.get_l_comb();
        fp.lm_coeffs = rgt.get_lm_coeffs();
    }

    reg_coeffs = yaml.get_coeffs();
    for (auto& c: reg_coeffs) c *= 2.0;

    fp.params = yaml.get_pair_params();
    fp.params_conditional = yaml.get_pair_params_conditional();

    mass = yaml.get_mass();
    const bool icharge = false;
    // type_indices (optional)
    vector1i type_indices = yaml.get_type_indices();

}
