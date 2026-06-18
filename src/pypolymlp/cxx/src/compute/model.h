/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

****************************************************************************/

#ifndef __MODEL
#define __MODEL

#include <Eigen/Dense>
#include "mlpcpp.h"
#include "polymlp/polymlp_api.h"
#include "compute/local.h"
#include "compute/local_pair.h"
#include "compute/neighbor_full.h"


class Model {

    PolymlpAPI polymlp;
    std::vector<std::vector<PolynomialTerm> > terms1, terms2, terms3;

    void pair(
        NeighborFull& neigh,
        const vector1i& types,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    void gtinv(
        NeighborFull& neigh,
        const vector1i& types,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    void model_polynomial(
        const Eigen::VectorXd& de,
        const Eigen::MatrixXd& df,
        const Eigen::MatrixXd& ds,
        const int type1,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    void model_order1(
        const PolynomialTerm& term,
        const Eigen::VectorXd& de,
        const Eigen::MatrixXd& df,
        const Eigen::MatrixXd& ds,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    void model_order2(
        const PolynomialTerm& term,
        const Eigen::VectorXd& de,
        const Eigen::MatrixXd& df,
        const Eigen::MatrixXd& ds,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    void model_order3(
        const PolynomialTerm& term,
        const Eigen::VectorXd& de,
        const Eigen::MatrixXd& df,
        const Eigen::MatrixXd& ds,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    void reshape(
        const vector1d& de,
        const vector2d& dfx,
        const vector2d& dfy,
        const vector2d& dfz,
        const vector2d& ds,
        const bool force,
        Eigen::VectorXd& de_eig,
        Eigen::MatrixXd& df_eig,
        Eigen::MatrixXd& ds_eig);


    public:

    Model();
    Model(const struct feature_params& fp);
    ~Model();

    void run(
        NeighborFull& neigh,
        const vector1i& types,
        const bool force,
        Eigen::VectorXd& xe_sum,
        Eigen::MatrixXd& xf_sum,
        Eigen::MatrixXd& xs_sum);

    int get_n_features();

};

#endif
