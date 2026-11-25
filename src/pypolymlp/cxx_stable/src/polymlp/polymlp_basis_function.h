/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

 ******************************************************************************/

#ifndef __POLYMLP_BASIS_FUNCTION_HPP
#define __POLYMLP_BASIS_FUNCTION_HPP

#include "polymlp_mlpcpp.h"


constexpr double pi = 3.1415926535897932384626433832795;

inline double cosine_cutoff_function(const double dis, const double cutoff){
    if (dis < cutoff)
        return 0.5 * (cos (pi * dis / cutoff) + 1.0);
    return 0.0;
}
inline void cosine_cutoff_function_d(
    const double dis, const double cutoff, double& cf, double& cf_d){
    if (dis < cutoff){
        double val1 = pi / cutoff;
        double val2 = val1 * dis;
        cf = 0.5 * (cos(val2) + 1.0);
        cf_d = - 0.5 * val1 * sin(val2);
    }
    else {
        cf = 0.0;
        cf_d = 0.0;
    }
}

inline double gauss(double x, double beta, double mu) {
    double dx = x - mu;
    return std::exp(-beta * dx * dx);
}
inline void gauss_d(double dis, double beta, double mu, double& bf, double& bf_d){
    double dx = dis - mu;
    bf = std::exp(-beta * dx * dx);
    bf_d = -2.0 * beta * dx * bf;
}


/********** Deprecated **********/
double bump_cutoff_function(const double& dis, const double& cutoff);

void bump_cutoff_function_d(const double& dis,
                            const double& cutoff,
                            double& bf,
                            double& bf_d);
double cosine(const double& dis, const double& param1);
void cosine_d(const double& dis,
              const double& param1,
              double& bf,
              double& bf_d);
double sine(const double& dis, const double& param1);
double polynomial(const double& dis, const double& param1);
void polynomial_d(const double& dis,
                  const double& param1,
                  double& bf,
                  double& bf_d);
double exp1(const double& dis, const double& param1);
double exp2(const double& dis, const double& param1);
double exp3(const double& dis, const double& param1);
void exp1_d(const double& dis, const double& param1, double& bf, double& bf_d);
void exp2_d(const double& dis, const double& param1, double& bf, double& bf_d);
void exp3_d(const double& dis, const double& param1, double& bf, double& bf_d);
double sto(const double& dis, const double& param1, const double& param2);
void sto_d(const double& dis,
           const double& param1,
           const double& param2,
           double& bf,
           double& bf_d);
double gto(const double& dis, const double& param1, const double& param2);
void gto_d(const double& dis,
           const double& param1,
           const double& param2,
           double& bf,
           double& bf_d);
double morlet(const double& dis, const double& param1);
void morlet_d(const double& dis,
              const double& param1,
              double& bf,
              double& bf_d);

double modified_morlet(const double& dis, const double& param1);
void modified_morlet_d(const double& dis,
                       const double& param1,
                       double& bf,
                       double& bf_d);

double mexican(const double& dis, const double& param1);
void mexican_d(const double& dis,
               const double& param1,
               double& bf,
               double& bf_d);

double morse(const double& dis, const double& param1, const double& param2);
void morse_d(const double& dis,
             const double& param1,
             const double& param2,
             double& bf,
             double& bf_d);
#endif
