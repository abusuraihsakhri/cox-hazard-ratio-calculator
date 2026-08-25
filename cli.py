#!/usr/bin/env python3
"""
CLI for Cox Proportional Hazards Model.

Usage:
  python cli.py fit --time 1 2 3 4 5 --event 1 0 1 1 0 --covariate 0.5 1.2 0.8 1.5 0.3
  python cli.py compare --time 1 2 3 4 5 --event 1 0 1 1 0 --covariate 0 0 1 1 1
  python cli.py batch --input sample.csv --output results.csv
"""

import argparse
import json
import sys

from cox_hr import cox_ph, hazard_ratio, forest_plot_data, check_proportional_hazards, summary, process_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cox-hr",
        description="Cox Proportional Hazards Model",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- fit ---
    p_fit = sub.add_parser("fit", help="Fit Cox PH model")
    p_fit.add_argument("--time", type=float, nargs="+", required=True, help="Observed times")
    p_fit.add_argument("--event", type=int, nargs="+", required=True, help="Event indicators (1=event, 0=censored)")
    p_fit.add_argument("--covariate", type=float, nargs="+", required=True, help="Covariate values (single covariate)")
    p_fit.add_argument("--json", action="store_true", help="Output as JSON")

    # --- multivariate ---
    p_multi = sub.add_parser("multivariate", help="Fit multivariate Cox PH model")
    p_multi.add_argument("--time", type=float, nargs="+", required=True)
    p_multi.add_argument("--event", type=int, nargs="+", required=True)
    p_multi.add_argument("--covariates", type=str, required=True,
                         help="JSON array of covariate arrays, e.g. '[[0.5,1],[1.2,0],[0.8,1]]'")
    p_multi.add_argument("--labels", type=str, nargs="+", help="Covariate labels")
    p_multi.add_argument("--json", action="store_true")

    # --- ph-check ---
    p_ph = sub.add_parser("ph-check", help="Check proportional hazards assumption")
    p_ph.add_argument("--time", type=float, nargs="+", required=True)
    p_ph.add_argument("--event", type=int, nargs="+", required=True)
    p_ph.add_argument("--covariate", type=float, nargs="+", required=True)

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Process CSV file")
    p_batch.add_argument("-i", "--input", required=True, help="Input CSV path")
    p_batch.add_argument("-o", "--output", default="results.csv", help="Output CSV path")

    args = parser.parse_args(argv)

    if args.command == "fit":
        covariates = [[x] for x in args.covariate]
        if args.json:
            result = cox_ph(args.time, args.event, covariates)
            print(json.dumps(result, indent=2, default=str))
        else:
            print(summary(args.time, args.event, covariates))
        return 0

    elif args.command == "multivariate":
        covariates = json.loads(args.covariates)
        if args.json:
            result = cox_ph(args.time, args.event, covariates)
            print(json.dumps(result, indent=2, default=str))
        else:
            print(summary(args.time, args.event, covariates, labels=args.labels))
        return 0

    elif args.command == "ph-check":
        covariates = [[x] for x in args.covariate]
        result = check_proportional_hazards(args.time, args.event, covariates)
        print("Proportional Hazards Assumption Check")
        print("=" * 50)
        for check in result["checks"]:
            status = "PASS" if check["assumption_holds"] else "FAIL"
            print(f"  Covariate {check['covariate']}: r={check['correlation']:.4f}, "
                  f"p={check['p_value']:.6f} [{status}]")
        return 0

    elif args.command == "batch":
        result = process_csv(args.input, args.output)
        print(f"Processed {result['n_subjects']} subjects, {result['n_events']} events -> {args.output}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
