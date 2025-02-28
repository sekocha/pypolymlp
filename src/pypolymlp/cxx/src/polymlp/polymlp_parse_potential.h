/****************************************************************************

        Copyright (C) 2025 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#ifndef __POLYMLP_PARSE_POTENTIAL
#define __POLYMLP_PARSE_POTENTIAL

#include "polymlp_mlpcpp.h"
#include "polymlp_read_gtinv.h"

std::string replace(std::string target, std::string str1, std::string str2);
vector1str split(const std::string& s, char delim);
vector1str split_list(const std::string& s);

void parse_polymlp(
    const char *file,
    feature_params& fp,
    vector1d& reg_coeffs,
    std::vector<std::string>& ele,
    vector1d& mass
);

class ParsePolymlpYaml {

    std::unordered_map<std::string, vector1str> params;
    vector2str pair_params;
    vector3str pair_params_conditional;

    void assign_exceptional_parameters(std::ifstream& ifs, const std::string& key);
    vector2str parse_2d(std::ifstream& ifs, const int n_lines);

    vector1i transform_vector1i(vector1str& strings);
    vector1d transform_vector1d(vector1str& strings);

    public:

    ParsePolymlpYaml();
    ParsePolymlpYaml(const char *file);
    ~ParsePolymlpYaml();

    const vector1str& get_elements();
    const double get_cutoff();
    const std::string& get_pair_type();
    const std::string& get_feature_type();
    const int get_max_p();
    const int get_max_l();
    const int get_model_type();

    const int get_gtinv_order();
    const int get_gtinv_version();
    const vector1i get_gtinv_maxl();
    const vector1b get_gtinv_sym();

    const vector1d get_mass();
    const vector1d get_coeffs();
    const vector2d get_pair_params();
    const vector3i get_pair_params_conditional();

    const int get_type_full();
    const vector1i get_type_indices();
};


#endif
