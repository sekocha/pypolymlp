/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_products.h"

void compute_products_real(
    const vector2i& map, const vector1dc& element, vector1d& prod_vals
){
    prod_vals = vector1d(map.size());
    int idx(0);
    for (const auto& prod: map){
        prod_vals[idx] = compute_product_real(prod, element);
        ++idx;
    }
}

double compute_product_real(const vector1i& prod, const vector1dc& element){

    const size_t n = prod.size();
    if (n == 3)
        return prod_real(element[prod[0]] * element[prod[1]], element[prod[2]]);
    if (n == 2) return prod_real(element[prod[0]], element[prod[1]]);
    if (n == 1) return element[prod[0]].real();
    if (n == 0) return 1.0;

    dc val_p = element[prod[1]];
    for (size_t i = 2; i < n; ++i) val_p *= element[prod[i]];
    return prod_real(val_p, element[prod[0]]);
}

double prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

dc prod_real_and_complex(const double val1, const dc& val2){
    return dc(val1 * val2.real(), val1 * val2.imag());
}
