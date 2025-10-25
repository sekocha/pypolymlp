/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __POLYMLP_PRODUCTS
#define __POLYMLP_PRODUCTS

#include "polymlp_mlpcpp.h"

template<typename T>
T compute_product(const vector1i& prod, const std::vector<T>& element){

    const size_t n = prod.size();
    if (n == 0) return 1.0;

    T val_p = element[prod[0]];
    for (size_t i = 1; i < n; ++i) val_p *= element[prod[i]];
    return val_p;
}


template<typename T>
void compute_products(
    const vector2i& map, const std::vector<T>& element, std::vector<T>& prod_vals
){
    prod_vals = std::vector<T>(map.size());
    int idx(0);
    for (const auto& prod: map){
        prod_vals[idx] = compute_product<T>(prod, element);
        ++idx;
    }
}


double compute_product_real(const vector1i& prod, const vector1dc& element);
void compute_products_real(
    const vector2i& map,
    const vector1dc& element,
    vector1d& prod_vals
);


double prod_real(const dc& val1, const dc& val2);
dc prod_real_and_complex(const double val1, const dc& val2);

#endif
