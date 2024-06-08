/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

 ******************************************************************************/

#include "polymlp_basis_function.h"

const double pi(3.141592653589793);

double cosine_cutoff_function(const double& dis, const double& cutoff){
    if (dis < cutoff)
        return 0.5 * (cos (pi * dis / cutoff) + 1.0);
    else
        return 0.0;
}
double cosine_cutoff_function_d(const double& dis, const double& cutoff){
    if (dis < cutoff)
        return - 0.5 * pi / cutoff * sin (pi * dis / cutoff);
    else
        return 0.0;
}
double bump_cutoff_function(const double& dis, const double& cutoff){
    double x = dis / cutoff;
    return exp (- 1.0 / (1.0 - x*x));
}
void bump_cutoff_function_d(const double& dis,
                            const double& cutoff,
                            double& bf,
                            double& bf_d){
    double x = dis / cutoff;
    double tmp1 = 1.0 - pow(x, 20);
    double tmp2 = pow(x, 19);
    bf = exp (- 1.0 / tmp1);
    bf_d = - 20 * tmp2 * bf / (tmp1 * tmp1);
}
double gauss(const double& x, const double& beta, const double& mu){
    return exp(- beta * pow (fabs(x - mu), 2.0));
}
void gauss_d(const double& dis,
             const double& p1,
             const double& p2,
             double& bf,
             double& bf_d){
    bf = gauss(dis, p1, p2);
    bf_d = - 2.0 * p1 * (dis - p2) * bf;
}
double cosine(const double& dis, const double& p1){
    return cos(p1 * dis);
}
void cosine_d(const double& dis,
              const double& p1,
              double& bf,
              double& bf_d){
    bf = cosine(dis, p1);
    bf_d = - p1 * sin(p1 * dis);
}
double sine(const double& dis, const double& p1){
    return sin(p1 * dis);
}

double polynomial(const double& dis, const double& param1){
    return pow(dis, param1);
}
void polynomial_d
(const double& dis, const double& param1, double& bf, double& bf_d){
    bf = polynomial(dis, param1);
    if (fabs(param1) < 1e-15)
        bf_d = 0.0;
    else
        bf_d = param1 * pow(dis, param1-1.0);
}

double exp1(const double& dis, const double& param1){
    return exp(-param1 * dis);
}
double exp2(const double& dis, const double& param1){
    return exp(-param1 * dis) / dis;
}
double exp3(const double& dis, const double& param1){
    return exp(-param1 * dis) * dis;
}
void exp1_d(const double& dis, const double& param1, double& bf, double& bf_d){
    bf = exp1(dis, param1);
    bf_d = - param1 * bf;
}
void exp2_d(const double& dis, const double& param1, double& bf, double& bf_d){
    bf = exp2(dis, param1);
    bf_d = - (1.0/dis + param1) * bf;
}
void exp3_d(const double& dis, const double& param1, double& bf, double& bf_d){
    bf = exp3(dis, param1);
    bf_d = (1.0/dis - param1) * bf;
}
double sto(const double& dis, const double& param1, const double& param2){
    return pow(dis, param1) * exp(-param2 * dis);
}
void sto_d(const double& dis,
           const double& param1,
           const double& param2,
           double& bf,
           double& bf_d){
    bf = sto(dis, param1, param2);
    bf_d = (param1 - param2 * dis) * bf / dis;
}
double gto(const double& dis, const double& param1, const double& param2){
    return pow(dis, param1) * gauss(param2, dis, 0.0);
}
void gto_d(const double& dis,
           const double& param1,
           const double& param2,
           double& bf,
           double& bf_d){
    bf = gto(dis, param1, param2);
    bf_d = (param1 - 2.0 * param2 * dis * dis) * bf / dis;
}
double morlet(const double& dis, const double& param1){
    return gauss(0.5, dis, param1) - gauss(0.5, -dis, param1);
}
void morlet_d(const double& dis,
              const double& param1,
              double& bf,
              double& bf_d){
    bf = morlet(dis, param1);
    bf_d = (param1 - dis) * gauss(0.5, dis, param1)
        + (param1 + dis) * gauss(0.5, -dis, param1);
}
double modified_morlet(const double& dis, const double& param1){
    return cos(param1 * dis) / cosh(dis);
}
void modified_morlet_d(const double& dis,
                       const double& param1,
                       double& bf,
                       double& bf_d){
    bf = modified_morlet(dis, param1);
    bf_d = - param1 * sin(param1 * dis) / cosh(dis) - bf * tanh(dis);
}
double mexican(const double& dis, const double& param1){
    double prod1 = pow (dis, 2.0) * pow (param1, 2.0);
    double prod2 = exp(- 0.5 * prod1);
    return sqrt(param1) * (1.0 - prod1) * prod2;
}
void mexican_d(const double& dis,
               const double& param1,
               double& bf,
               double& bf_d){
    double prod1 = pow (dis, 2.0) * pow (param1, 2.0);
    double prod2 = exp(- 0.5 * prod1);
    double prod3 = dis * pow (param1, 2.5);
    bf = sqrt(param1) * (1.0 - prod1) * prod2;
    bf_d = - prod3 * (3.0 - prod1) * prod2;
}
double morse(const double& dis, const double& param1, const double& param2){
    double prod1 = - param1 * (dis - param2);
    return exp(2.0 * prod1) - 2.0 * exp(prod1);
}
void morse_d(const double& dis,
             const double& param1,
             const double& param2,
             double& bf,
             double& bf_d){
    double prod1 = - param1 * (dis - param2);
    bf =  exp(2.0 * prod1) - 2.0 * exp(prod1);
    bf_d = - 2.0 * param1 * (bf + exp(prod1));
}

/*
double bessel(const double& dis, const double& param1){
    int index = round(param1);
    return boost::math::cyl_bessel_j(index, dis);
}
double neumann(const double& dis, const double& p1){
    return boost::math::cyl_neumann(round(p1), dis);
}
double sph_bessel(const double& dis, const double& p1, const double& p2){
    double pdis = p2 * dis;
    return boost::math::sph_bessel(round(p1), pdis);
}
double sph_neumann(const double& dis, const double& p1){
    return boost::math::sph_neumann(round(p1), dis);
}
void bessel_d(const double& dis, const double& p1, double& bf, double& bf_d){

    int index = round(p1);
    bf = boost::math::cyl_bessel_j(index, dis);
    if (index == 0)
        bf_d = - boost::math::cyl_bessel_j(1, dis);
    else {
        double prod1 = boost::math::cyl_bessel_j(index-1, dis)
            - boost::math::cyl_bessel_j(index+1, dis);
        bf_d = prod1 * 0.5;
    }
}
void neumann_d(const double& dis, const double& p1, double& bf, double& bf_d){

    int index = round(p1);
    bf = boost::math::cyl_neumann(index, dis);
    if (index == 0)
        bf_d = - boost::math::cyl_neumann(1, dis);
    else {
        double prod1 = boost::math::cyl_neumann(index-1, dis)
            - boost::math::cyl_neumann(index+1, dis);
        bf_d = prod1 * 0.5;
    }
}
void sph_bessel_d(const double& dis,
                  const double& p1,
                  const double& p2,
                  double& bf,
                  double& bf_d){

//  accuracy of derivatives may be not good, particularly n > 10.
    int index = round(p1);
    double pdis = p2 * dis;
    bf = boost::math::sph_bessel(index, pdis);
    if (index == 0) bf_d = - boost::math::sph_bessel(1, pdis) * p2;
    else {
        bf_d = boost::math::sph_bessel(index-1, pdis)
            - (index + 1) * bf / (pdis);
        bf_d *= p2;
    }

}
void sph_neumann_d(const double& dis,
                   const double& p1,
                   double& bf,
                   double& bf_d){

    int index = round(p1);
    bf = boost::math::sph_neumann(index, dis);
    if (index == 0) bf_d = - boost::math::sph_neumann(1, dis);
    else bf_d = boost::math::sph_neumann(index-1, dis) - (index + 1) * bf / dis;
}
*/
