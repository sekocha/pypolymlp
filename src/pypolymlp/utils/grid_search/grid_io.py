"""Functions for saving parameters to files."""

from pypolymlp.core.data_format import PolymlpParamsSingle


def save_params(params: PolymlpParamsSingle, filename: str = "polymlp.in"):
    """Save MLP parameters to a file."""
    model = params.model
    with open(filename, "w") as f:
        print("feature_type", model.feature_type, file=f)
        print("cutoff", model.cutoff, file=f)
        print("n_gaussians", model.n_gaussians, file=f)

        alpha = params.regression_alpha
        print("reg_alpha_params", alpha[0], alpha[1], alpha[2], file=f)
        print(file=f)

        print("model_type", model.model_type, file=f)
        print("max_p", model.max_p, file=f)

        if model.feature_type == "gtinv":
            gtinv = model.gtinv
            print("gtinv_order", gtinv.order, file=f)
            print("gtinv_maxl", end="", file=f)
            for maxl in gtinv.max_l:
                print("", maxl, end="", file=f)
            print("", file=f)
        print(file=f)

        print("include_force", params.include_force, file=f)
        print("include_stress", params.include_stress, file=f)
        print(file=f)

        print("elements", end="", file=f)
        for ele in params.elements:
            print("", ele, end="", file=f)
        print("", file=f)
