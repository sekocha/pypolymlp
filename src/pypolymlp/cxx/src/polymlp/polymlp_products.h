/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_PRODUCTS
#define __POLYMLP_PRODUCTS

#include "polymlp_mlpcpp.h"

template<typename T>
void compute_products(
    const vector2i& map, const std::vector<T>& element, std::vector<T>& prod_vals
){

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

void compute_products_real(const vector2i& map, const vector1dc& element, vector1d& prod_vals);

double prod_real(const dc& val1, const dc& val2);
dc prod_real_and_complex(const double val1, const dc& val2);

#endif
