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

#ifndef __LOCAL
#define __LOCAL

#include <cmath>

#include "mlpcpp.h"
#include "polymlp/polymlp_functions_interface.h"
#include "polymlp/polymlp_model_params.h"

class Local{

    int n_atom, atom1, type1, n_fn, n_des, n_type; 
    struct feature_params fp;
    ModelParams modelp;

    vector3dc compute_anlm(const vector2d& dis_a, 
                           const vector3d& diff_a, 
                           const vector2i& lm_info);
    void compute_anlm_d(const vector2d& dis_a, 
                        const vector3d& diff_a, 
                        const vector2i& atom2_a, 
                        const vector2i& lm_info, 
                        vector3dc& anlm, 
                        vector4dc& anlm_dfx, 
                        vector4dc& anlm_dfy, 
                        vector4dc& anlm_dfz, 
                        vector4dc& anlm_ds);

    public: 

    Local();
    Local(const int& n_atom_i, 
          const int& atom1_i, 
          const int& type1_i,
          const struct feature_params& fp_i, 
          const ModelParams& modelp);
    ~Local();

    vector1d pair(const vector2d& dis_a);
    void pair_d(const vector2d& dis_a, 
                const vector3d& diff_a, 
                const vector2i& atom2_a,
                vector1d& an, 
                vector2d& an_dfx, 
                vector2d& an_dfy, 
                vector2d& an_dfz, 
                vector2d& an_ds);

    vector1d gtinv(const vector2d& dis_a, const vector3d& diff_a);
    void gtinv_d(const vector2d& dis_a, 
                 const vector3d& diff_a, 
                 const vector2i& atom2_a,
                 vector1d& dn, 
                 vector2d& dn_dfx, 
                 vector2d& dn_dfy, 
                 vector2d& dn_dfz, 
                 vector2d& dn_ds);

    vector2i get_lm_info(const int& max_l);
};

#endif
