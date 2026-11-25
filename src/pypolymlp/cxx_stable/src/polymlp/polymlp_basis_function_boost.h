/****************************************************************************

  Copyright (C) 2024 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

******************************************************************************/

#ifndef __POLYMLP_BASIS_FUNCTION_BOOST_HPP
#define __POLYMLP_BASIS_FUNCTION_BOOST_HPP

#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/laguerre.hpp>

#include "polymlp_mlpcpp.h"

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

#endif
