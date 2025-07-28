"""Class for learning curves."""


def save_learning_curve_log(
    error_log: dict,
    filename: str = "polymlp_learning_curve.dat",
):
    """Save results from learning curve calculations."""
    f = open(filename, "w")
    header = "# n_str, RMSE(energy, meV/atom) RMSE(force, eV/ang), RMSE(stress)"
    print(header, file=f)
    for n_samp, error in error_log:
        print(
            n_samp,
            error["energy"] * 1000,
            error["force"],
            error["stress"],
            file=f,
        )
    f.close()


def print_learning_curve_log(error_log: dict):
    """Generate output for results from learning curve calculations."""
    print("Learning Curve:", flush=True)
    for n_samples, error in error_log:
        print("- n_samples:   ", n_samples, flush=True)
        print(
            "  rmse_energy: ",
            "{:.8f}".format(error["energy"] * 1000),
            flush=True,
        )
        print("  rmse_force:  ", "{:.8f}".format(error["force"]), flush=True)
        print("  rmse_stress: ", error["stress"], flush=True)
