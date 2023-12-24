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

#include "polymlp_basis_function_boost.h"

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


