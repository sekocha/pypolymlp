"""Functions for saving parameters to files."""

from pypolymlp.core.data_format import PolymlpParams


def save_params(params: PolymlpParams, filename: str = "polymlp.in"):
    """Save MLP parameters to a file."""
    model = params.model
    f = open(filename, "w")
    print("feature_type", model.feature_type, file=f)
    print("cutoff", model.cutoff, file=f)
    g1, g2 = model.pair_params_in1, model.pair_params_in2
    print("gaussian_params1", g1[0], g1[1], g1[2], file=f)
    print("gaussian_params2", g2[0], g2[1], g2[2], file=f)

    alpha = params.regression_alpha
    print("reg_alpha_params", alpha[0], alpha[1], alpha[2], file=f)
    print("", file=f)

    print("model_type", model.model_type, file=f)
    print("max_p", model.max_p, file=f)

    if model.feature_type == "gtinv":
        gtinv = model.gtinv
        print("gtinv_order", gtinv.order, file=f)
        print("gtinv_maxl", end="", file=f)
        for maxl in gtinv.max_l:
            print("", maxl, end="", file=f)
        print("", file=f)
    print("", file=f)

    print("include_force", params.include_force, file=f)
    print("include_stress", params.include_stress, file=f)

    f.close()


# def write_grid_hybrid(params_grid1, params_grid2, iseq=0):
#
#     grid_pairs = itertools.product(params_grid1, params_grid2)
#     for params_dict1, params_dict2 in grid_pairs:
#         idx = str(iseq + 1).zfill(5)
#         dirname = "model_grid/polymlp-hybrid-" + idx + "/"
#         os.makedirs(dirname, exist_ok=True)
#         write_params_dict(params_dict1, dirname + "polymlp.in")
#         write_params_dict(params_dict2, dirname + "polymlp.in.2")
#         iseq += 1
#
#     return iseq
