/****************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __MODELFORCHARGEPY
#define __MODELFORCHARGEPY

#include <iomanip>

#include "mlpcpp.h"
#include "compute/neighbor.h"
#include "compute/compute_features.h"

#include <Eigen/Core>

class ModelForChargePy{

    Eigen::MatrixXd x_all;
    vector1i xc_begin_idx_dataset;

    public:

    ModelForChargePy(const vector3d& axis,
                     const vector3d& positions_c,
                     const vector2i& types,
                     const int& n_type,
                     const vector2d& params,
                     const double& cutoff,
                     const std::string& pair_type,
                     const std::string& des_type,
                     const int& model_type,
                     const int& maxp,
                     const int& maxl,
                     const vector3i& lm_array,
                     const vector2i& l_comb,
                     const vector2d& lm_coeffs,
                     const vector1i& n_st_dataset,
                     const bool& print_memory);

    ~ModelForChargePy();

    Eigen::MatrixXd& get_x();
    const vector1i& get_cbegin() const;

};

#endif
