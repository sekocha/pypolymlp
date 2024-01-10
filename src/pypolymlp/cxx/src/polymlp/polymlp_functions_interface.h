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

****************************************************************************/

#ifndef __POLYMLP_FUNCTIONS_INTERFACE
#define __POLYMLP_FUNCTIONS_INTERFACE

#include "polymlp_mlpcpp.h"
#include "polymlp_basis_function.h"
#include "polymlp_mlipkk_spherical_harmonics.h"

//#include <boost/geometry.hpp>
//namespace bg = boost::geometry;
//namespace bm = boost::math;

// Radial functions
void get_fn_(const double& dis, 
             const struct feature_params& fp, 
             vector1d& fn);

void get_fn_(const double& dis, 
             const struct feature_params& fp, 
             vector1d& fn, 
             vector1d& fn_dr);

// Spherical harmonics
void get_ylm_(const double polar, 
              const double azimuthal, 
              const int lmax, 
              vector1dc& ylm);

void get_ylm_(const double r, 
              const double polar, 
              const double azimuthal, 
              const int lmax, 
              vector1dc& ylm, 
              vector1dc& ylm_dx, 
              vector1dc& ylm_dy, 
              vector1dc& ylm_dz);

vector1d cartesian_to_spherical_(const vector1d& v);

#endif
