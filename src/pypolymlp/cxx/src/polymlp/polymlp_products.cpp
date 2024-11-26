/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#include "polymlp_products.h"

void compute_products_real(const vector2i& map, const vector1dc& element, vector1d& prod_vals){

    prod_vals = vector1d(map.size());

    int idx(0);
    dc val_p;
    for (const auto& prod: map){
        if (prod.size() > 1) {
            auto iter = prod.begin() + 1;
            val_p = element[*iter];
            ++iter;
            while (iter != prod.end()){
                val_p *= element[*iter];
                ++iter;
            }
            prod_vals[idx] = prod_real(val_p, element[*(prod.begin())]);
        }
        else if (prod.size() == 1){
            prod_vals[idx] = element[*(prod.begin())].real();
        }
        else prod_vals[idx] = 1.0;
        ++idx;
    }
}

double prod_real(const dc& val1, const dc& val2){
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

dc prod_real_and_complex(const double val1, const dc& val2){
    return dc(val1 * val2.real(), val1 * val2.imag());
}
