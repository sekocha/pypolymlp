#!/usr/bin/env python
import yaml


def print2d(array, varname):

    print("    " + varname + " =")
    print("        {{" + ", ".join(array[0]) + "},")
    for a1 in array[1:-1]:
        print("        {" + ", ".join(a1) + "},")
    print("        {" + ", ".join(array[-1]) + "}};")
    print("")


def print2dnewline(array, varname):

    print("    " + varname + " =")
    print("        {{" + ",\n        ".join(array[0]) + "},")
    for a1 in array[1:-1]:
        print("        {" + ",\n         ".join(a1) + "},")
    print("        {" + ",\n         ".join(array[-1]) + "}};")
    print("")


def print3d(array, varname):

    print("    " + varname + " = {")
    for i, a2 in enumerate(array):
        if len(a2) == 1:
            print("        {{" + ", ".join(a2[0]) + "}},")
        else:
            print("        {{" + ", ".join(a2[0]) + "},")
            for a1 in a2[1:-1]:
                print("        {" + ", ".join(a1) + "},")
            if i == len(array) - 1:
                print("        {" + ", ".join(a2[-1]) + "}}};")
            else:
                print("        {" + ", ".join(a2[-1]) + "}},")
    print("")


l_all_orders, m_all_orders, c_all_orders = [], [], []
# maxl = [[1],[30],[10,12,20],[2,4,8],[2],[2]]
maxl = [[1], [30], [12], [4], [2], [1]]
for order in range(1, 7):
    l_all, m_all, c_all = [], [], []
    yamlfile = "lists_ver2/basis-order" + str(order) + "-l0.yaml"
    f = open(yamlfile)
    yamldata = yaml.safe_load(f)
    f.close()

    for j in range(len(maxl[order - 1])):
        if j == 0:
            minl_t, maxl_t = 0, maxl[order - 1][j]
        else:
            minl_t, maxl_t = maxl[order - 1][j - 1] + 1, maxl[order - 1][j]

        for d in yamldata["basis_set"]:
            lcomb = d["lcomb"]
            if sum(lcomb) % 2 == 0 and max(lcomb) <= maxl_t and max(lcomb) >= minl_t:
                l_all.append([str(l1) for l1 in lcomb])
                mcombs = [mc_attr[0] for mc_attr in d["mcombs_coeffs"]]
                coeffs = [str(mc_attr[1]) for mc_attr in d["mcombs_coeffs"]]
                m_all.append([[str(m2) for m2 in m1] for m1 in mcombs])
                c_all.append(coeffs)

    l_all_orders.append(l_all)
    m_all_orders.append(m_all)
    c_all_orders.append(c_all)

for i, (l_all, m_all, c_all) in enumerate(
    zip(l_all_orders, m_all_orders, c_all_orders)
):
    order = str(i + 1)
    print("void GtinvDataVer2::set_gtinv_info_" + order + "(){")
    print("")

    print2d(l_all, "l_array_all_" + order)
    print2dnewline(c_all, "coeffs_all_" + order)
    print3d(m_all, "m_array_all_" + order)

    print("}")


"""
def print2d(array, varname):

    print('    '+varname+' =')
    print('        {{'+', '.join(array[0])+'},', end='')
    for a1 in array[1:-1]:
        print('{'+', '.join(a1)+'},', end='')
    print('{'+', '.join(array[-1])+'}};')
    print('')


def print2dnewline(array, varname):

    print('    '+varname+' =')
    print('        {{'+','.join(array[0])+'},', end='')
    for a1 in array[1:-1]:
        print('{'+','.join(a1)+'},', end='')
    print('{'+','.join(array[-1])+'}};')
    print('')


def print3d(array, varname):

    print('    '+varname+' = {')
    for i, a2 in enumerate(array):
        if i == 0:
            print('       ', end='')
        if len(a2) == 1:
            print('{{'+', '.join(a2[0])+'}},', end='')
        else:
            print('{{'+', '.join(a2[0])+'},', end='')
            for a1 in a2[1:-1]:
                print('{'+', '.join(a1)+'},', end='')
            if i == len(array)-1:
                print('{'+', '.join(a2[-1])+'}}};')
            else:
                print('{'+', '.join(a2[-1])+'}},', end='')
    print('')
"""
