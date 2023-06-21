/*****************************************************************************

        Copyright (C) 2011 Atsuto Seko
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

	    Header file for ewald.cpp

******************************************************************************/

#ifndef __EWALD
#define __EWALD

#include "mlpcpp.h"
#include "compute/neighbor.h"

class Ewald{

    int n_atom;
    double eta, er, eg, eself, eall, correction;
    arma::mat charge_prod;
    arma::vec f, s, fr, fg, sr, sg;
    vector1d fvec, svec, frvec, fgvec, srvec, sgvec;

    double realspace(const vector3d& dis_array, 
                     const vector3i& atom2_array);
    double reciprocal(const vector2d& gvectors, 
                      const vector2d& positions_cartesian, 
                      const double& volume);
    double realspace_f(const vector3d& dis_array, 
                       const vector4d& diff_array,
                       const vector3i& atom2_array, 
                       arma::vec& fr, 
                       arma::vec& sr);
    double reciprocal_f(const vector2d& gvectors, 
                        const vector2d& positions_cartesian, 
                        const double& volume, 
                        arma::vec& fg, 
                        arma::vec& sg);
    double self();

    vector2vec get_diff_array(const vector2d& positions_c);
    vector1d get_gcoeff(const vector2d& gvectors);

    template<typename T>
        T dforce_to_dstress
        (const T& dforce, const arma::vec& diff_c);

    public: 

    Ewald();
    Ewald(const vector2d& axis, 
          const vector2d& positions_c,
          const vector1i& types, 
          const int& n_type,
          const double& cutoff,
          const vector2d& gvectors,
          const vector1d& charge,
          const double& volume,
          const double& eta_i,
          const bool& force);

    ~Ewald();

    const double& get_real_energy() const;
    const double& get_reciprocal_energy() const;
    const double& get_self_energy() const;
    const double& get_energy() const;
    const arma::vec& get_force() const;
    const arma::vec& get_real_force() const;
    const arma::vec& get_reciprocal_force() const;
    const arma::vec& get_stress() const;
    const arma::vec& get_real_stress() const;
    const arma::vec& get_reciprocal_stress() const;
    const vector1d& get_force_vector1d() const;
    const vector1d& get_real_force_vector1d() const;
    const vector1d& get_reciprocal_force_vector1d() const;
    const vector1d& get_stress_vector1d() const;
    const vector1d& get_real_stress_vector1d() const;
    const vector1d& get_reciprocal_stress_vector1d() const;

};

#endif
