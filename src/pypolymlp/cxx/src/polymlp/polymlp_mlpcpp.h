/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#ifndef __POLYMLP
#define __POLYMLP

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <array>
#include <string>
#include <complex>
#include <numeric>
#include <algorithm>
#include <iterator>

using vector1i = std::vector<int>;
using vector2i = std::vector<vector1i>;
using vector3i = std::vector<vector2i>;
using vector4i = std::vector<vector3i>;
using vector1d = std::vector<double>;
using vector2d = std::vector<vector1d>;
using vector3d = std::vector<vector2d>;
using vector4d = std::vector<vector3d>;
using dc = std::complex<double>;
using vector1dc = std::vector<dc>;
using vector2dc = std::vector<vector1dc>;
using vector3dc = std::vector<vector2dc>;
using vector4dc = std::vector<vector3dc>;

template<typename T>
void print_time(clock_t& start, clock_t& end, const T& memo){

    std::cout << " elapsed time: " << memo << ": "
        << (double)(end-start) / CLOCKS_PER_SEC << " (sec.)" << std::endl;

}

struct feature_params {
    int n_type;
    bool force;
    vector2d params;
    double cutoff;
    std::string pair_type;
    std::string des_type;
    int model_type;
    int maxp;
    int maxl;
    vector3i lm_array;
    vector2i l_comb;
    vector2d lm_coeffs;
};

// Hash function must be examined
class HashVI {
    public:
        size_t operator()(const std::vector<int> &x) const {
            const int C = 997;
            size_t t = 0;
            for (size_t i = 0; i != x.size(); ++i) {
                t = t * C + x[i];
            }
            return t;
        }
};


#endif
