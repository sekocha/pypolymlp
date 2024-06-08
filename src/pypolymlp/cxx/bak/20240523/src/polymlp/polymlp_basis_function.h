/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

 ******************************************************************************/

#ifndef __POLYMLP_BASIS_FUNCTION_HPP
#define __POLYMLP_BASIS_FUNCTION_HPP

#include "polymlp_mlpcpp.h"

/*
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/laguerre.hpp>
*/

double cosine_cutoff_function(const double& dis, const double& cutoff);

double cosine_cutoff_function_d(const double& dis, const double& cutoff);

double bump_cutoff_function(const double& dis, const double& cutoff);

void bump_cutoff_function_d(const double& dis,
                            const double& cutoff,
                            double& bf,
                            double& bf_d);

double gauss(const double& x, const double& beta, const double& mu);

void gauss_d(const double& dis,
             const double& param1,
             const double& param2,
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
/*
double bessel(const double& dis, const double& param1);

double neumann(const double& dis, const double& param1);

double sph_bessel(const double& dis, const double& p1, const double& p2);

double sph_neumann(const double& dis, const double& param1);

void bessel_d(const double& dis,
              const double& param1,
              double& bf,
              double& bf_d);

void neumann_d(const double& dis,
               const double& param1,
               double& bf,
               double& bf_d);

void sph_bessel_d(const double& dis,
                  const double& param1,
                  const double& param2,
                  double& bf,
                  double& bf_d);

void sph_neumann_d(const double& dis,
                   const double& param1,
                   double& bf,
                   double& bf_d);
*/

#endif
