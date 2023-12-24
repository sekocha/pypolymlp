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

#ifndef __NEIGHBOR
#define __NEIGHBOR

#include "mlpcpp.h"

class Neighbor{

    vector3d dis_array; 
    vector4d diff_array; 
    vector3i atom2_array;

    vector2d find_trans(const vector2d& axis, const double& cutoff);
    vector1d prod(const vector2d& mat, const vector1i& vec);

    public: 

    Neighbor
        (const vector2d& axis, const vector2d& positions_c, 
         const vector1i& types, const int& n_type, const double& cutoff);

    ~Neighbor();

    const vector3d& get_dis_array() const;
    const vector4d& get_diff_array() const;
    const vector3i& get_atom2_array() const;

};

#endif
