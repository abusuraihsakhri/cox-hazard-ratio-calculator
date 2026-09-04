#!/usr/bin/env python3
"""
CLI for Cox Proportional Hazards Model.

Usage:
  python cli.py fit --time 1 2 3 4 5 --event 1 0 1 1 0 --covariate 0.5 1.2 0.8 1.5 0.3
  python cli.py multivariate --time 1 2 3 4 5 --event 1 0 1 1 0 --covariates '[[0.5,65],[1.2,70]]'
  python cli.py ph-check --time 1 2 3 4 5 --event 1 0 1 1 0 --covariate 0.5 1.2 0.8 1.5 0.3
  python cli.py batch --input sample.csv --output results.csv
  python cli.py audit --task-id TASK-01 --target KEY-01 --primary 12.0 --secondary 4.0 --status NOMINAL
  python cli.py chat Explain specifications
  python cli.py verify-audit
"""

import argparse
import json
import os
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

    # --- audit ---
    p_audit = sub.add_parser("audit", help="Run supervisor audit on a task payload")
    p_audit.add_argument("--task-id", required=True, help="Task identifier")
    p_audit.add_argument("--target", default="KEY-001", help="Target identifier")
    p_audit.add_argument("--primary", type=float, default=12.0, help="Primary metric value")
    p_audit.add_argument("--secondary", type=float, default=4.0, help="Secondary metric value")
    p_audit.add_argument("--status", default="NOMINAL", help="Status descriptor")
    p_audit.add_argument("--critical", action="store_true", help="Mark as critical")

    # --- chat ---
    p_chat = sub.add_parser("chat", help="Supervisory chat interface")
    p_chat.add_argument("message", nargs="+", help="Chat message words")

    # --- verify-audit ---
    sub.add_parser("verify-audit", help="Verify HMAC audit trail integrity")

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

    elif args.command == "audit":
        from agents.supervisor import SystemSupervisor
        from agents.models import SystemTaskPayload
        supervisor = SystemSupervisor(model_provider=os.getenv("MODEL_PROVIDER", "mock"))
        payload = SystemTaskPayload(
            task_id=args.task_id,
            target_identifier=args.target,
            primary_metric=args.primary,
            secondary_metric=args.secondary,
            status_descriptor=args.status,
            is_critical_flag=args.critical,
        )
        dossier = supervisor.process_task(payload)
        print(json.dumps(dossier.to_dict(), indent=2, default=str))
        return 0

    elif args.command == "chat":
        from agents.supervisor import SystemSupervisor
        supervisor = SystemSupervisor(model_provider=os.getenv("MODEL_PROVIDER", "mock"))
        message = " ".join(args.message)
        response = supervisor.query_supervisory_chat(message)
        print(response)
        return 0

    elif args.command == "verify-audit":
        from agents.base import AuditLogger
        verified = AuditLogger.verify_integrity()
        trail = AuditLogger.get_trail()
        print(f"Audit Trail Integrity: {'VERIFIED' if verified else 'COMPROMISED'}")
        print(f"Total audit blocks: {len(trail)}")
        return 0 if verified else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
