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
