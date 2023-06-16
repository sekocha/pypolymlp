#!/usr/bin/env python
import numpy as np
from polymlp_generator.common.table import mass_table

def print_param(dict1, key, fstream, prefix=''):
    print(str(dict1[key]), '#', prefix + key, file=fstream)

def print_array1d(array, fstream, comment='', fmt=None):
    for obj in array:
        if fmt is not None:
            print(fmt.format(obj), end=' ', file=fstream)
        else:
            print(obj, end=' ', file=fstream)
    print('#', comment, file=fstream)

def save_mlp_lammps(params_dict, coeffs, scales, elements,
                        filename='mlp.lammps'):

    f = open(filename, 'w')
    print_array1d(elements, f, comment='elements')
    model_dict = params_dict['model']
    print_param(model_dict, 'cutoff', f)
    print_param(model_dict, 'pair_type', f)
    print_param(model_dict, 'feature_type', f)
    print_param(model_dict, 'model_type', f)
    print_param(model_dict, 'max_p', f)
    print_param(model_dict, 'max_l', f)

    if model_dict['feature_type'] == 'gtinv':
        gtinv_dict = model_dict['gtinv']
        print_param(gtinv_dict, 'order', f, prefix='gtinv_')
        print_array1d(gtinv_dict['max_l'], f, comment='gtinv_max_l')
        gtinv_sym = [0 for _ in gtinv_dict['max_l']]
        print_array1d(gtinv_sym, f, comment='gtinv_sym')

    print(len(coeffs), '# n_coeffs', file=f)
    print_array1d(coeffs, f, comment='reg. coeffs', fmt="{0:15.15e}")
    print_array1d(scales, f, comment='scales', fmt="{0:15.15e}")

    print(len(model_dict['pair_params']), '# n_params', file=f)
    for obj in model_dict['pair_params']:
        print("{0:15.15f}".format(obj[0]), "{0:15.15f}".format(obj[1]), 
              '# pair func. params', file=f)
        
    mass = [mass_table()[ele] for ele in elements]
    print_array1d(mass, f, comment='atomic mass', fmt="{0:15.15e}")
    print('False # electrostatic', file=f)
    f.close()

    def _read_var(self, f, dtype=int, return_list=False):

        line = f.readline()
        l = line.split('#')[0].split()
        if return_list == True:
            return [dtype(v) for v in l]
        return dtype(l[0])

    def read_pot_for_lammps(self, filename='mlp.lammps'):
        
        f = open(filename)

        self.di = DataInput()
        self.elements = self._read_var(f, str, return_list=True)
        self.di.model_e.cutoff = self._read_var(f, float)
        self.di.model_e.pair_type = self._read_var(f, str)
        self.di.model_e.des_type = self._read_var(f, str)
        self.di.model_e.model_type = self._read_var(f)
        self.di.model_e.maxp = self._read_var(f)
        self.di.model_e.maxl = self._read_var(f)

        if (self.di.model_e.des_type == 'gtinv'):
            self.di.model_e.gtinv_order = self._read_var(f)
            self.di.model_e.gtinv_maxl = self._read_var(f, return_list=True)
            self.di.model_e.gtinv_sym = self._read_var(f, strtobool, 
                                                       return_list=True)

        n_coeffs = self._read_var(f)
        self.coeffs = self._read_var(f, float, return_list=True)
        self.scales = self._read_var(f, float, return_list=True)

        n_des_params = self._read_var(f)
        self.di.model_e.des_params = []
        for n in range(n_des_params):
            params = self._read_var(f, float, return_list=True)
            self.di.model_e.des_params.append(params)

        self.mass = self._read_var(f, float, return_list=True)
        self.di.es = self._read_var(f, strtobool)
        if self.di.es == True:
            print(' not implemented for self.di.es = True')

        f.close()

        self.di.n_type = len(self.elements)
        self.read_gtinv()

        return self

    def read_gtinv(self):

        rgi = mlpcpp.Readgtinv(self.di.model_e.gtinv_order, 
                               self.di.model_e.gtinv_maxl, 
                               self.di.model_e.gtinv_sym, 
                               self.di.n_type)
        self.di.model_e.lm_seq = rgi.get_lm_seq()
        self.di.model_e.l_comb = rgi.get_l_comb()
        self.di.model_e.lm_coeffs = rgi.get_lm_coeffs()

        return self


'''
class Pot:
    def __init__(self,
                 coefs=None,
                 scale=None,
                 reg=None,
                 scaler=None,
                 rmse=None,
                 di:DataInput=None,
                 elements=None):

        if coefs is not None:
            self.reg = Ridge()
            self.reg.intercept_ = 0.0
            self.reg.coef_ = coefs
        else:
            self.reg = reg

        if scale is not None:
            self.scaler = StandardScaler(with_mean=False)
            self.scaler.scale_ = scale
        else:
            self.scaler = scaler

        self.scaled_coeff = self.reg.coef_ / self.scaler.scale_

        self.rmse = rmse
        self.di = di
        self.elements = elements
        if elements is not None:
            self.mass = [mass_table()[ele] for ele in elements]

    def save_pot(self, file_name='mlp.pkl'):
        joblib.dump(self, file_name, compress=3)

    def _read_var(self, f, dtype=int, return_list=False):

        line = f.readline()
        l = line.split('#')[0].split()
        if return_list == True:
            return [dtype(v) for v in l]
        return dtype(l[0])

    def read_pot_for_lammps(self, filename='mlp.lammps'):
        
        f = open(filename)

        self.di = DataInput()
        self.elements = self._read_var(f, str, return_list=True)
        self.di.model_e.cutoff = self._read_var(f, float)
        self.di.model_e.pair_type = self._read_var(f, str)
        self.di.model_e.des_type = self._read_var(f, str)
        self.di.model_e.model_type = self._read_var(f)
        self.di.model_e.maxp = self._read_var(f)
        self.di.model_e.maxl = self._read_var(f)

        if (self.di.model_e.des_type == 'gtinv'):
            self.di.model_e.gtinv_order = self._read_var(f)
            self.di.model_e.gtinv_maxl = self._read_var(f, return_list=True)
            self.di.model_e.gtinv_sym = self._read_var(f, strtobool, 
                                                       return_list=True)

        n_coeffs = self._read_var(f)
        self.coeffs = self._read_var(f, float, return_list=True)
        self.scales = self._read_var(f, float, return_list=True)

        n_des_params = self._read_var(f)
        self.di.model_e.des_params = []
        for n in range(n_des_params):
            params = self._read_var(f, float, return_list=True)
            self.di.model_e.des_params.append(params)

        self.mass = self._read_var(f, float, return_list=True)
        self.di.es = self._read_var(f, strtobool)
        if self.di.es == True:
            print(' not implemented for self.di.es = True')

        f.close()

        self.di.n_type = len(self.elements)
        self.read_gtinv()

        return self

    def read_gtinv(self):

        rgi = mlpcpp.Readgtinv(self.di.model_e.gtinv_order, 
                               self.di.model_e.gtinv_maxl, 
                               self.di.model_e.gtinv_sym, 
                               self.di.n_type)
        self.di.model_e.lm_seq = rgi.get_lm_seq()
        self.di.model_e.l_comb = rgi.get_l_comb()
        self.di.model_e.lm_coeffs = rgi.get_lm_coeffs()

        return self

    def get_structural_features_multiple(self, st_array, print_memory=False):

        terms_obj = Features(st_array, 
                             self.di, 
                             [len(st_array)], 
                             [False], 
                             print_memory=print_memory)
        return np.array(terms_obj.get_x())

    def get_structural_features(self, 
                                st=None, 
                                print_memory=False,
                                file_poscar='POSCAR'):
        if st is None:
            st = Poscar(file_poscar).get_structure_class()
        terms_obj = FeaturesSingle(st, self.di, print_memory=print_memory)
        return np.array(terms_obj.get_x())

    def predict(self, X1, scaled_x=False):

        if scaled_x == False:
            coeff = self.scaled_coeff
        else:
            coeff = self.reg.coef_

        # temporarily multi-threaded using numba,
        pred = numba_support.mat_dot_vec(X1, coeff)
        return pred

        # this np.dot is not multi-threaded, which must be improved.
        #return np.dot(X1, coeff)
        #return np.reshape(self.dot_blas(X1, coeff1), -1)

#    def dot_blas(self, A, B):
#        dot = scipy.linalg.get_blas_funcs('gemm', (A, B))
#        return dot(alpha=1.0, a=A, b=B)

    def compute_rmse_from_features(self, X, y_true, 
                                   normalize=[],
                                   return_y=False, 
                                   scaled_x=False, 
                                   weight_x=1.0, 
                                   copy=True):

        if abs(weight_x - 1.0) > 1e-8:
            if copy == True:
                X1 = X / weight_x
            else:
                numba_support.mat_divide(X, weight_x)
                X1 = X
        else:
            X1 = X

        y_pred = self.predict(X1, scaled_x=scaled_x)

        if len(normalize) == len(y_true):
            normalize = np.array(normalize)
            #y_true /= normalize 
            #y_pred /= normalize 
            y_true = numba_support.vec_divide_vec(y_true, normalize)
            y_pred = numba_support.vec_divide_vec(y_pred, normalize)

        if return_y == False:
            return np.sqrt(np.mean(np.square(y_true - y_pred)))
        else:
            return np.sqrt(np.mean(np.square(y_true - y_pred))), y_true, y_pred

    def property(self, 
                 st=None, 
                 file_poscar='POSCAR', 
                 stress_unit='GPa',
                 force=True, 
                 print_time=False, 
                 return_time=False):

        if st is None:
            p = Poscar(file_poscar)
            axis, positions, n_atoms, elements, types = p.get_structure()
            st = Structure(axis, positions, n_atoms, elements, types)
        volume = st.get_volume()
        self.di.wforce = force

        terms_obj = Features([st], self.di, [1], [force])
        x = terms_obj.get_x()
        fbegin, sbegin = terms_obj.get_fbegin()[0], terms_obj.get_sbegin()[0]

        self.xe, self.xf, self.xs = x[0], x[fbegin:], x[sbegin:fbegin]

        #########################################
        # computing charges (should be updated)
        #########################################
        if self.di.es == True:
            if self.di.charge_model == False:
                charge = [c for c,n in \
                    zip(self.di.charge_atom, st.n_atoms) for j in range(n)]
            else:
                pass

        if (force == False):
            e = self.reg.predict(self.scaler.transform([self.xe]))[0]
            if self.di.es == True:
                _,_,_,e_es = electrostatic(st,charge,wfactor=1e-2,force=False)
                e += e_es
            return e 
        else:
            e = self.reg.predict(self.scaler.transform([self.xe]))[0]
            f = self.reg.predict(self.scaler.transform(self.xf))
            s_ev = self.reg.predict(self.scaler.transform(self.xs))
            print(e)
            if self.di.es == True:
                esr,esg,ess,e_es,f_es,s_es \
                    = electrostatic(st,charge,wfactor=1e-2)
                print(e_es)
                e += e_es
                f += f_es
                s_ev += s_es

            f = np.reshape(f, (-1,3))
            s_gpa = s_ev / volume * 160.21766208
            return e, f, s_gpa, s_ev

    def rotate_check(self, file_poscar='POSCAR', theta=1.0):

        e1, _, _, _ = self.property(file_poscar=file_poscar)

        p = Poscar(file_poscar)
        axis, positions, n_atoms, elements, types = p.get_structure()
        rot = np.array([[cos(theta), -sin(theta), 0], \
            [sin(theta), cos(theta), 0], [0, 0, 1]])
        st = Structure(np.dot(rot,axis), positions, n_atoms, elements, types)
        e2, _, _, _ = self.property(st=st)

        return e2-e1
 
    def numerical_force(self, file_poscar='POSCAR', eps=1e-5): # eps: angstrom

        e1, f1, _, _ = self.property(file_poscar=file_poscar)

        p = Poscar(file_poscar)
        axis, positions, n_atoms, elements, types = p.get_structure()
        axis_inv = np.linalg.inv(axis)

        disp_c = np.array([[eps, 0, 0], [0, eps, 0], [0, 0, eps]])
        disp_f = np.dot(axis_inv, disp_c)

        f_array = []
        for i in range(positions.shape[1]):
            tmp = []
            for j in range(3):
                positions_new = positions.copy()
                positions_new[:,i] += disp_f[:,j]
                st = Structure(axis, positions_new, n_atoms, elements, types)
                e2, _, _, _ = self.property(st=st)
                tmp.append(-(e2 - e1)/eps)
            f_array.append(tmp)
        f2 = np.array(f_array)

        return f2-f1, f1, f2

    def numerical_stress(self, file_poscar='POSCAR', eps=1e-5): # eps: strain 

        e1, _, _, s1 = self.property(file_poscar=file_poscar)
        print(e1)

        p = Poscar(file_poscar)
        axis, positions, n_atoms, elements, types = p.get_structure()
        
        s2 = []
        for index in [[0,0], [1,1], [2,2], [0,1], [1,2], [0,2]]:
            expand = np.identity(3)
            expand[index[0], index[1]] += eps
            axis_new = np.dot(expand, axis)
            st = Structure(axis_new, positions, n_atoms, elements, types)
            e2, _, _, _ = self.property(st=st)
            print(e2)
            s2.append(-(e2-e1)/eps)
        s2 = np.array(s2)

        return s2-s1, s1, s2
 

    def get_pot(self):
        return self.reg, self.scaler, self.rmse, self.di

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pot', type=str, \
        default='mlp.pkl', help='Python compressed file for potential')
    parser.add_argument('--poscar', type=str, default='poscar', \
        help='poscar file name')
    parser.add_argument('--numerical_force', action='store_true', \
        help='Numerical force mode')
    parser.add_argument('--rotate_check', action='store_true', \
        help='Rotational invariance mode')
    parser.add_argument('--time', action='store_true', \
        help='Output computational time')
    args = parser.parse_args()

    pot = joblib.load(args.pot)

    if (args.time==True):
        a = time.time()

    if (args.numerical_force==True):
        np.set_printoptions(formatter={'float': '{:3.10g}'.format})
        diff2, s1, s2 = pot.numerical_stress\
            (file_poscar=args.poscar, eps=1e-7)
        print(' analytical stress = ') 
        print(s1)
        print(' numerical stress = ') 
        print(s2)
        print(' diff (stress) = ') 
        print(diff2)
        diff, f1, f2 = pot.numerical_force(file_poscar=args.poscar, eps=1e-6)
        print(' analytical force = ') 
        print(f1)
        print(' numerical force = ') 
        print(f2)
        print(' diff (force) = ') 
        print(diff)
    elif (args.rotate_check==True):
        diff = pot.rotate_check(file_poscar=args.poscar, theta=0.7)
        print(diff)
    else:
        e, f, s_gpa, s_ev = pot.property(file_poscar=args.poscar)
        print(' energy = ', e)
        print(' force = ')
        print(f)
        print(' stress = ')
        print('  xx: ', s_gpa[0], '(GPa)')
        print('  yy: ', s_gpa[1], '(GPa)')
        print('  zz: ', s_gpa[2], '(GPa)')
        print('  xy: ', s_gpa[3], '(GPa)')
        print('  yz: ', s_gpa[4], '(GPa)')
        print('  zx: ', s_gpa[5], '(GPa)')
        print('  xx: ', s_ev[0], '(eV)')
        print('  yy: ', s_ev[1], '(eV)')
        print('  zz: ', s_ev[2], '(eV)')
        print('  xy: ', s_ev[3], '(eV)')
        print('  yz: ', s_ev[4], '(eV)')
        print('  zx: ', s_ev[5], '(eV)')

    if (args.time==True):
        b = time.time()
        print(' elapsed time =', b-a, '(s)')
'''

