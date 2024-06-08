/*****************************************************************************

    Copyright (C) 2020 Atsuto Seko
    seko@cms.mtl.kyoto-u.ac.jp

    Class for calculating electrostatic energy using Ewald method

 ******************************************************************************/

#include "ewald.h"

const double pi = 3.14159265358979323846;

Ewald::Ewald(){}

Ewald::Ewald(const vector2d& axis,
             const vector2d& positions_c,
             const vector1i& types,
             const int& n_type,
             const double& cutoff,
             const vector2d& gvectors,
             const vector1d& charge,
             const double& volume,
             const double& eta_i,
             const bool& force):eta(eta_i){

    Neighbor neigh(axis, positions_c, types, n_type, cutoff);
    const auto &dis_array = neigh.get_dis_array();
    const auto &diff_array = neigh.get_diff_array();
    const auto &atom2_array = neigh.get_atom2_array();

    const double coulomb_electron = 1.602176634e-19;
    //correction = coulomb_electron * 1e10 / (4.0 * pi * 8.8541878128e-12);
    correction = coulomb_electron * 1e10 * 8.9875517923e9;

    n_atom = types.size();

    charge_prod = arma::mat(n_atom, n_atom, arma::fill::zeros);
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        for (int atom2 = 0; atom2 <= atom1; ++atom2){
            charge_prod(atom1,atom2) = charge[atom1] * charge[atom2];
        }
    }
    charge_prod = arma::symmatl(charge_prod);

    if (force == false) {
        er = realspace(dis_array, atom2_array);
        eg = reciprocal(gvectors, positions_c, volume);
    }
    else {
    /*
        f = arma::vec(3*n_atom, arma::fill::zeros);
        s = arma::vec(6, arma::fill::zeros);
        er = realspace_f(dis_array, diff_array, atom2_array, f, s);
        eg = reciprocal_f(gvectors, positions_c, volume, f, s);
        fvec = arma::conv_to<vector1d>::from(f);
        svec = arma::conv_to<vector1d>::from(s);
    */
        fr = arma::vec(3*n_atom, arma::fill::zeros);
        fg = arma::vec(3*n_atom, arma::fill::zeros);
        sr = arma::vec(6, arma::fill::zeros);
        sg = arma::vec(6, arma::fill::zeros);
        er = realspace_f(dis_array, diff_array, atom2_array, fr, sr);
        eg = reciprocal_f(gvectors, positions_c, volume, fg, sg);
        f = fr + fg;
        s = sr + sg;
        fvec = arma::conv_to<vector1d>::from(f);
        svec = arma::conv_to<vector1d>::from(s);
        frvec = arma::conv_to<vector1d>::from(fr);
        srvec = arma::conv_to<vector1d>::from(sr);
        fgvec = arma::conv_to<vector1d>::from(fg);
        sgvec = arma::conv_to<vector1d>::from(sg);
    }
    eself = self();
    eall = er + eg + eself;
}

Ewald::~Ewald(){}

double Ewald::realspace(const vector3d& dis_array,
                        const vector3i& atom2_array){

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    double er1 = 0.0;
    for (int atom1 = 0; atom1 < dis_array.size(); ++atom1){
        for (int t2 = 0; t2 < dis_array[atom1].size(); ++t2){
            for (int j = 0; j < dis_array[atom1][t2].size(); ++j){
                double dis = dis_array[atom1][t2][j];
                int atom2 = atom2_array[atom1][t2][j];
                er1 += charge_prod(atom1,atom2) * erfc(sqrt(eta) * dis) / dis;
            }
        }
    }
    return 0.5 * correction * er1;
}

double Ewald::reciprocal(const vector2d& gvectors,
                         const vector2d& positions_c,
                         const double& volume){

    vector2vec diff = get_diff_array(positions_c);
    vector1d gcoeff = get_gcoeff(gvectors);

    double eg1 = 0.0, trace = arma::trace(charge_prod);
    for (int g = 0; g < gvectors.size(); ++g){
        arma::vec gvec = arma::vec(gvectors[g]);
        double eg2 = 0.0;
        for (int atom1 = 0; atom1 < n_atom; ++atom1){
            for (int atom2 = 0; atom2 < atom1; ++atom2){
                double g_prod_r = arma::dot(gvec, diff[atom1][atom2]);
                eg2 += charge_prod(atom1,atom2) * cos(g_prod_r);
            }
        }
        eg1 += (eg2 + eg2 + trace) * gcoeff[g];
    }
    return eg1 * (2 * pi / volume) * correction;
}

double Ewald::realspace_f(const vector3d& dis_array,
                          const vector4d& diff_array,
                          const vector3i& atom2_array,
                          arma::vec& fr,
                          arma::vec& sr){

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    int atom2;
    double er1 = 0.0, dis, c1, coeff1;
    arma::vec fr1(3*n_atom, arma::fill::zeros),
        sr1(6, arma::fill::zeros), dr_dx, tmp;
    for (int atom1 = 0; atom1 < diff_array.size(); ++atom1){
        for (int t2 = 0; t2 < diff_array[atom1].size(); ++t2){
            for (int j = 0; j < diff_array[atom1][t2].size(); ++j){
                dis = dis_array[atom1][t2][j];
                dr_dx = arma::vec(diff_array[atom1][t2][j]) / dis;
                atom2 = atom2_array[atom1][t2][j];

                c1 = erfc(sqrt(eta) * dis) / dis;
                er1 += charge_prod(atom1,atom2) * c1;
                coeff1 = c1 + 2 * sqrt(eta) * exp(-eta * dis * dis) / sqrt(pi);
                coeff1 *= - charge_prod(atom1, atom2) / dis;
                tmp = coeff1 * dr_dx;
                fr1.rows(atom1*3,atom1*3+2) += tmp;
                fr1.rows(atom2*3,atom2*3+2) -= tmp;
                sr1 -= dforce_to_dstress
                    (tmp, arma::vec(diff_array[atom1][t2][j]));
            }
        }
    }
    double coeff = 0.5 * correction;
    fr += coeff * fr1;
    sr += coeff * sr1;
    return coeff * er1;
}

double Ewald::reciprocal_f(const vector2d& gvectors,
                           const vector2d& positions_c,
                           const double& volume,
                           arma::vec& fg,
                           arma::vec& sg){

    vector2vec diff = get_diff_array(positions_c);
    vector1d gcoeff = get_gcoeff(gvectors);

    double eg1 = 0.0, eg2, g_prod_r, trace = arma::trace(charge_prod);
    arma::vec fg1(3*n_atom, arma::fill::zeros),
        sg1(6, arma::fill::zeros), gvec, tmp;
    for (int g = 0; g < gvectors.size(); ++g){
        gvec = arma::vec(gvectors[g]);
        eg2 = 0.0;
        arma::vec fg2(3*n_atom, arma::fill::zeros);
        for (int atom1 = 0; atom1 < n_atom; ++atom1){
            for (int atom2 = 0; atom2 < atom1; ++atom2){
                g_prod_r = arma::dot(gvec, diff[atom1][atom2]);
                eg2 += charge_prod(atom1,atom2) * cos(g_prod_r);
                tmp = - 2.0 * charge_prod(atom1,atom2) * sin(g_prod_r) * gvec;
                fg2.rows(atom1*3,atom1*3+2) += tmp;
                fg2.rows(atom2*3,atom2*3+2) -= tmp;
            }
        }
        double lambda2 = 2.0 * (1/(4*eta)+1/arma::dot(gvec,gvec));
        double prod = (eg2 + eg2 + trace) * gcoeff[g];
        arma::vec tensor_vec(6);
        for (int i = 0; i < 3; ++i){
            tensor_vec(i) = 1 - gvec[i]*gvec[i]*lambda2;
        }
        tensor_vec(3) = - gvec[0]*gvec[1]*lambda2;
        tensor_vec(4) = - gvec[1]*gvec[2]*lambda2;
        tensor_vec(5) = - gvec[0]*gvec[2]*lambda2;
        eg1 += prod;
        fg1 += fg2 * gcoeff[g];
        sg1 += prod * tensor_vec;
    }

    double coeff = 2 * pi * correction / volume;
    fg += coeff * fg1;
    sg += coeff * sg1;
    return coeff * eg1;
}


double Ewald::self(){
    return -arma::trace(charge_prod) * sqrt(eta/pi) * correction;
}

vector2vec Ewald::get_diff_array(const vector2d& positions_c){

    vector2vec diff(n_atom, vector1vec(n_atom));
    for (int atom1 = 0; atom1 < n_atom; ++atom1){
        for (int atom2 = 0; atom2 < atom1; ++atom2){
            arma::vec tmp(3);
            tmp(0) = positions_c[0][atom2] - positions_c[0][atom1];
            tmp(1) = positions_c[1][atom2] - positions_c[1][atom1];
            tmp(2) = positions_c[2][atom2] - positions_c[2][atom1];
            diff[atom1][atom2] = tmp;
        }
    }
    return diff;
}

vector1d Ewald::get_gcoeff(const vector2d& gvectors){

    vector1d gcoeff;
    for (arma::vec gvec: gvectors){
        double square_g = arma::dot(gvec, gvec);
        gcoeff.emplace_back(exp(-square_g / (4 * eta)) / square_g);
    }
    return gcoeff;
}

template<typename T>
T Ewald::dforce_to_dstress
(const T& dforce, const arma::vec& diff_c){

    T dstress(6);
    for (int i = 0; i < 3; ++i)
        dstress(i) = dforce(i) * diff_c(i);
    dstress(3) = dforce(0) * diff_c(1);
    dstress(4) = dforce(1) * diff_c(2);
    dstress(5) = dforce(2) * diff_c(0);

    return dstress;
}

const double& Ewald::get_real_energy() const{ return er; }
const double& Ewald::get_reciprocal_energy() const{ return eg; }
const double& Ewald::get_self_energy() const{ return eself; }
const double& Ewald::get_energy() const{ return eall; }
const arma::vec& Ewald::get_force() const{ return f; }
const arma::vec& Ewald::get_real_force() const{ return fr; }
const arma::vec& Ewald::get_reciprocal_force() const{ return fg; }
const arma::vec& Ewald::get_stress() const{ return s; }
const arma::vec& Ewald::get_real_stress() const{ return sr; }
const arma::vec& Ewald::get_reciprocal_stress() const{ return sg; }
const vector1d& Ewald::get_force_vector1d() const{ return fvec; }
const vector1d& Ewald::get_real_force_vector1d() const{ return frvec; }
const vector1d& Ewald::get_reciprocal_force_vector1d() const{ return fgvec; }
const vector1d& Ewald::get_stress_vector1d() const{ return svec; }
const vector1d& Ewald::get_real_stress_vector1d() const{ return srvec; }
const vector1d& Ewald::get_reciprocal_stress_vector1d() const{ return sgvec; }
