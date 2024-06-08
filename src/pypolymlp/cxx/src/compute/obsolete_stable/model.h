/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __MODEL
#define __MODEL

#include <omp.h>
#include <algorithm>

#include "mlpcpp.h"
#include "compute/local.h"
#include "polymlp/polymlp_model_params.h"

class Model{

    int n_atom, n_type, model_type, maxp;
    bool force;
    ModelParams modelp;

    vector1i types;

    vector1d xe_sum;
    vector2d xf_sum, xs_sum;

    void pair(const vector3d& dis_array_all,
              const vector4d& diff_array_all,
              const vector3i& atom2_array_all,
              const struct feature_params& fp);

    void gtinv(const vector3d& dis_array,
               const vector4d& diff_array,
               const vector3i& atom2_array,
               const struct feature_params& fp);

    void model_common(const vector1d& de,
                      const vector2d& dfx,
                      const vector2d& dfy,
                      const vector2d& dfz,
                      const vector2d& ds,
                      const int& type1);
    void model_linear(const vector1d& de,
                      const vector2d& dfx,
                      const vector2d& dfy,
                      const vector2d& dfz,
                      const vector2d& ds,
                      int& col);
    void model1(const vector1d& de,
                const vector2d& dfx,
                const vector2d& dfy,
                const vector2d& dfz,
                const vector2d& ds,
                int& col);
    void model2_comb2(const vector1d& de,
                      const vector2d& dfx,
                      const vector2d& dfy,
                      const vector2d& dfz,
                      const vector2d& ds,
                      int& col);
    void model2_comb3(const vector1d& de,
                      const vector2d& dfx,
                      const vector2d& dfy,
                      const vector2d& dfz,
                      const vector2d& ds,
                      int& col);

    public:

    Model();
    Model(const vector3d& dis_array_all,
          const vector4d& diff_array_all,
          const vector3i& atom2_array_all,
          const vector1i& types_i,
          const struct feature_params& fp,
          const bool& element_swap);
    ~Model();

    const vector1d& get_xe_sum() const;
    const vector2d& get_xf_sum() const;
    const vector2d& get_xs_sum() const;

};

#endif
