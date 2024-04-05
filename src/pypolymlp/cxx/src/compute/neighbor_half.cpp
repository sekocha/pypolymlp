/****************************************************************************

        Copyright (C) 2024 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

*****************************************************************************/

#include "compute/neighbor_half.h"
#include <time.h>


NeighborHalf::NeighborHalf(const vector2d& axis, 
                           const vector2d& positions_c, 
                           const vector1i& types, 
                           const double& cutoff){

    clock_t start = clock();

    const double tol = 1e-12;
    const auto& trans = find_trans(axis, cutoff);
    clock_t start2 = clock();

    const int n_total_atom = types.size();
    half_list = vector2i(n_total_atom);
    diff_list = vector3d(n_total_atom);

    double dx, dy, dz;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j <= i; ++j){
            for (const auto& tr: trans){
                dx = positions_c[0][j] + tr[0] - positions_c[0][i];
                dy = positions_c[1][j] + tr[1] - positions_c[1][i];
                dz = positions_c[2][j] + tr[2] - positions_c[2][i];
                bool bool_half = false;
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    if (i == j){
                        if (dz >= tol) 
                            bool_half = true;
                        else if (fabs(dz) < tol and dy >= tol) 
                            bool_half = true;
                        else if (fabs(dz) < tol and fabs(dy) < tol and dx >=tol) 
                            bool_half = true;
                    }
                    else {
                        bool_half = true;
                    }
                }
                if (bool_half == true){
                    half_list[i].emplace_back(j);
                    diff_list[i].emplace_back(vector1d({dx, dy, dz}));
                }
            }
        }
    }

    clock_t end = clock();
    std::cout << (double)(start2 - start) / CLOCKS_PER_SEC << std::endl;
    std::cout << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
}

NeighborHalf::~NeighborHalf(){}

vector1d NeighborHalf::prod(const vector2d& mat, const vector1i& vec){

    vector1d res(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i){
        for (size_t j = 0; j < mat[i].size(); ++j){
            res[i] += mat[i][j] * vec[j];
        }
    }
    return res;
}

//vector2d NeighborHalf::find_trans(const vector2d& axis, const double& cutoff){
//
//    vector2i vertices;
//    for (int i = 0; i < 2; ++i){
//        for (int j = 0; j < 2; ++j){
//            for (int k = 0; k < 2; ++k){
//                vertices.emplace_back(vector1i{i,j,k});
//            }
//        }
//    }
// 
//    int m = 10;
//    vector2i cell_expansion;
//    for (int i = -m; i < m + 1; ++i){
//        for (int j = -m; j < m + 1; ++j){
//            for (int k = -m; k < m + 1; ++k){
//                vector1i vec = {i,j,k};
//                double min_dis(1e10);
//                for (const auto& vertex1: vertices){
//                    for (const auto& vertex2: vertices){
//                        vector1i dint = {vec[0] + vertex2[0] - vertex1[0],
//                                         vec[1] + vertex2[1] - vertex1[1],
//                                         vec[2] + vertex2[2] - vertex1[2]};
//                        vector1d vec_c = prod(axis, dint);
//                        double dis = sqrt(vec_c[0]*vec_c[0]
//                                +vec_c[1]*vec_c[1]
//                                +vec_c[2]*vec_c[2]);
//                        if (min_dis > dis){
//                            min_dis = dis;
//                        }
//                        if (min_dis < cutoff) break;
//                    }
//                    if (min_dis < cutoff) break;
//                }
//                if (min_dis < cutoff){
//                    cell_expansion.emplace_back(vec);
//                }
//
//                /*
//                vector1d vec_c = prod(axis, vec);
//                double dis = sqrt(vec_c[0]*vec_c[0]
//                                 +vec_c[1]*vec_c[1]
//                                 +vec_c[2]*vec_c[2]);
//                if (dis > 0 and dis < cutoff){
//                    double exp = ceil(cutoff / dis);
//                    for (int e = 0; e < exp + 1; ++e){
//                        cell_expansion.insert(vector1i{i*e, j*e, k*e});
//                    }
//                    cell_expansion.insert(vector1i{i*exp+1, j*exp+1, k*exp+1});
//
//                    //for (int l = 0; l < 3; ++l){
//                    //    if (exp * abs(vec[l]) > max_exp[l]){
//                    //        max_exp[l] = exp * abs(vec[l]);
//                    //    }
//                    //}
//                }
//                */
//            }
//        }
//    }
//
//    vector2d trans_c_array;
//    for (const auto& cell: cell_expansion){
//        trans_c_array.emplace_back(prod(axis, cell));
//    }
//    return trans_c_array;
//}

double NeighborHalf::distance(const vector2d& axis, 
                              const int i, const int j, const int k){
    vector1i vec = {i,j,k};
    vector1d vec_c = prod(axis, vec);
    double dis = sqrt(vec_c[0]*vec_c[0]
                    +vec_c[1]*vec_c[1]
                    +vec_c[2]*vec_c[2]);
    return dis;
}

vector2d NeighborHalf::find_trans(const vector2d& axis, const double& cutoff){

    vector2i vertices;
    for (int i = 0; i < 2; ++i){
        for (int j = 0; j < 2; ++j){
            for (int k = 0; k < 2; ++k){
                vertices.emplace_back(vector1i{i,j,k});
            }
        }
    }

    double max_length(0.0);
    for (const auto& ver1: vertices){
        for (const auto& ver2: vertices){
            double dis = distance(axis, 
                                  ver1[0] - ver2[0], 
                                  ver1[1] - ver2[1], 
                                  ver1[2] - ver2[2]);
            if (max_length < dis) max_length = dis;
        }
    }


    int m = 20;
    vector1i max_exp(3, 1);
    for (int i = -m; i < m + 1; ++i){
        for (int j = -m; j < m + 1; ++j){
            for (int k = -m; k < m + 1; ++k){
                double dis = distance(axis, i, j, k);
                if (dis > 0 and dis < cutoff){
                    double exp = ceil(cutoff / dis);
                    if (exp * abs(i) > max_exp[0]) max_exp[0] = exp * abs(i);
                    if (exp * abs(j) > max_exp[1]) max_exp[1] = exp * abs(j);
                    if (exp * abs(k) > max_exp[2]) max_exp[2] = exp * abs(k);
                }
            }
        }
    }
    for (int l = 0; l < 3; ++l) max_exp[l] += 1;

    vector2d trans_c_array;
    for (int i = -max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = -max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = -max_exp[2]; k < max_exp[2] + 1; ++k){
                vector1i vec = {i,j,k};
                vector1d vec_c = prod(axis, vec);
                double dis = sqrt(vec_c[0]*vec_c[0]
                                +vec_c[1]*vec_c[1]
                                +vec_c[2]*vec_c[2]);
                if (dis < max_length + cutoff){
                    trans_c_array.emplace_back(vec_c);
                }
            }
        }
    }
    return trans_c_array;
}

/*
vector2d NeighborHalf::find_trans(const vector2d& axis, const double& cutoff){

    int m = 10;
    vector1i max_exp(3,1);
    for (int i = 0; i < m + 1; ++i){
        for (int j = 0; j < m + 1; ++j){
            for (int k = 0; k < m + 1; ++k){
                vector1i vec = {i,j,k};
                vector1d vec_c = prod(axis, vec);
                double dis = sqrt(vec_c[0]*vec_c[0]
                    +vec_c[1]*vec_c[1]+vec_c[2]*vec_c[2]);
                if (dis > 0 and dis < cutoff * 2){
                    double exp = ceil((cutoff * 2)/dis);
                    for (int l = 0; l < 3; ++l){
                        if (exp * vec[l] > max_exp[l]) 
                            max_exp[l] = exp * vec[l];
                    }
                }
            }
        }
    }

    vector2d trans_c_array;
    for (int i = -max_exp[0]; i < max_exp[0] + 1; ++i){
        for (int j = -max_exp[1]; j < max_exp[1] + 1; ++j){
            for (int k = -max_exp[2]; k < max_exp[2] + 1; ++k){
                trans_c_array.emplace_back(prod(axis, vector1i{i,j,k}));
            }
        }
    }
    return trans_c_array;
}
*/


const vector2i& NeighborHalf::get_half_list() const{ return half_list; }
const vector3d& NeighborHalf::get_diff_list() const{ return diff_list; }


/* lammps convention for pair choice 
NeighborHalf::NeighborHalf(const vector2d& axis, 
                           const vector2d& positions_c, 
                           const vector1i& types, 
                           const double& cutoff){

    const double tol = 1e-12;
    const auto& trans = find_trans(axis, cutoff);

    const int n_total_atom = types.size();
    half_list = vector2i(n_total_atom);
    diff_list = vector3d(n_total_atom);

    double dx, dy, dz;
    for (int i = 0; i < n_total_atom; ++i){
        for (int j = 0; j < n_total_atom; ++j){
            for (const auto& tr: trans){
                dx = positions_c[0][j] + tr[0] - positions_c[0][i];
                dy = positions_c[1][j] + tr[1] - positions_c[1][i];
                dz = positions_c[2][j] + tr[2] - positions_c[2][i];
                bool bool_half = false;
                double dis = sqrt(dx*dx + dy*dy + dz*dz);
                if (dis < cutoff and dis > 1e-10){
                    if (dz >= tol) 
                        bool_half = true;
                    else if (fabs(dz) < tol and dy >= tol) 
                        bool_half = true;
                    else if (fabs(dz) < tol and fabs(dy) < tol and dx >=tol) 
                        bool_half = true;
                }
                if (bool_half == true){
                    half_list[i].emplace_back(j);
                    diff_list[i].emplace_back(vector1d({dx, dy, dz}));
                }
            }
        }
    }
}
*/


