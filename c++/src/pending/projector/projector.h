/****************************************************************************

        Copyright (C) 2017 Atsuto Seko
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

	    Header file for Projector.cpp
		
****************************************************************************/

#ifndef __PROJECTOR
#define __PROJECTOR

#include <gsl/gsl_sf_coupling.h>

#include "mlpcpp.h"

class Projector{

    vector1i row, col;
    vector1d data;

    int lm_to_matrix_index(const vector1i& l_list, const vector1i& m_array);

    vector2i mlist_nonzero(const vector1i& l_list, const vector2i& m_list);

    void order2(const vector1i& l_list, const vector2i& m_list);
    void order3(const vector1i& l_list, const vector2i& m_list);
    void order4(const vector1i& l_list, const vector2i& m_list);
    void order5(const vector1i& l_list, const vector2i& m_list);
    void order6(const vector1i& l_list, const vector2i& m_list);

    void array_initialize(const vector2i& m_list);

    public: 

    Projector();
    ~Projector();

    void build_projector(const vector1i& l_list, const vector2i& m_list);

    double clebsch_gordan
        (const int& l1, const int& l2, const int& l, 
         const int& m1, const int& m2, const int& m);

    const vector1i& get_row() const;
    const vector1i& get_col() const;
    const vector1d& get_data() const;
};

#endif
