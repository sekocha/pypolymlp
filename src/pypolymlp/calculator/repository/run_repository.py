#!/usr/bin/env python
import argparse

from pypolymlp.calculator.repository.repository_file_generation import (
    PolymlpRepositoryGeneration,
)
from pypolymlp.calculator.repository.repository_prediction import (
    PolymlpRepositoryPrediction,
)
from pypolymlp.calculator.repository.repository_web_contents import (
    PolymlpRepositoryWebContents,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="polymlp_summary_convex.yaml",
        help="Summary yaml file from grid search",
    )
    parser.add_argument(
        "--path_vasp",
        type=str,
        default="./",
        help="Path (vasp data for prototype structures)",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default="./",
        help="Path (output of predictions)",
    )
    parser.add_argument("--no_qha", action="store_false", help="QHA calculation")
    args = parser.parse_args()

    pred = PolymlpRepositoryPrediction(yamlfile=args.yaml, path_vasp=args.path_vasp)
    pred.run(path_output=args.path_output, run_qha=args.no_qha)

    rep_file = PolymlpRepositoryGeneration(path_data=args.path_output)
    rep_file.run()

    web = PolymlpRepositoryWebContents(path_data=args.path_output)
    web.run()
