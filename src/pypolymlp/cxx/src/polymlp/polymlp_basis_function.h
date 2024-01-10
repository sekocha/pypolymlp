/****************************************************************************

  Copyright (C) 2020 Atsuto Seko
  seko@cms.mtl.kyoto-u.ac.jp

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to
  the Free Software Foundation, Inc., 51 Franklin Street,
  Fifth Floor, Boston, MA 02110-1301, USA, or see
    http://www.gnu.org/copyleft/gpl.txt

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
