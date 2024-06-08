/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __MODEL_PROPERTIES
#define __MODEL_PROPERTIES

#include "mlpcpp.h"
#include "compute/local_fast.h"
#include "polymlp/polymlp_model_params.h"

class ModelProperties{

    int n_atom, n_type, model_type, maxp;
    bool use_force;
    ModelParams modelp;

    vector1i types;
    vector1d coeffs;

    double energy;
    vector1d force, stress;

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

    ModelProperties();
    ModelProperties(const vector3d& dis_array_all,
                    const vector4d& diff_array_all,
                    const vector3i& atom2_array_all,
                    const vector1i& types_i,
                    const vector1d& coeffs_i,
                    const struct feature_params& fp,
                    const bool& element_swap);
    ~ModelProperties();

    const double& get_energy() const;
    const vector1d& get_force() const;
    const vector1d& get_stress() const;

};

#endif
