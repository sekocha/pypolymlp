/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#ifndef __POLYMLP_PARSE_POTENTIAL_LEGACY
#define __POLYMLP_PARSE_POTENTIAL_LEGACY

#include "polymlp_mlpcpp.h"
#include "polymlp_read_gtinv.h"


bool check_polymlp_legacy(const char *file);
bool parse_elements(
    std::ifstream& input, std::vector<std::string>& ele
);

void parse_basic_params(std::ifstream& input, feature_params& fp);
int parse_gtinv_params(
    std::ifstream& input,
    vector1i& gtinv_maxl,
    std::vector<bool>& gtinv_sym
);
void parse_reg_coeffs(std::ifstream& input, vector1d& reg_coeffs);
int parse_gaussians(std::ifstream& input, feature_params& fp);
void parse_params_conditional(std::ifstream& input);
bool parse_type_indices(std::ifstream& input, vector1i& type_indices);

void parse_polymlp_legacy(
    const char *file,
    feature_params& fp,
    vector1d& reg_coeffs,
    std::vector<std::string>& ele,
    vector1d& mass
);

template<typename T>
T get_value(std::ifstream& input){

    std::string line;
    std::stringstream ss;

    T val;
    std::getline( input, line );
    ss << line;
    ss >> val;

    return val;
}

template<typename T>
T get_value(std::ifstream& input, std::string& tag){

    std::string line;
    std::stringstream ss;

    T val;
    std::getline( input, line );
    ss << line;
    ss >> val;
    ss >> tag;
    ss >> tag;

    return val;
}

template<typename T>
std::vector<T> get_value_array(std::ifstream& input, const int& size){

    std::string line;
    std::stringstream ss;

    std::vector<T> array(size);

    std::getline( input, line );
    ss << line;
    T val;
    for (int i = 0; i < array.size(); ++i){
        ss >> val;
        array[i] = val;
    }

    return array;
}

#endif
