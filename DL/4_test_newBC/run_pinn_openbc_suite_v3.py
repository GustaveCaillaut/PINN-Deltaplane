#!/usr/bin/env python3
"""Run the corrected open-boundary PINN experiments.

Examples
--------
Smoke test:
    python -u run_pinn_openbc_suite_v3.py --preset smoke --outdir runs_pinn_openbc_v3_smoke --no-save-model

Full pre-run:
    python -u run_pinn_openbc_suite_v3.py --preset full --outdir runs_pinn_openbc_v3_full

With XFOIL Cp data-prior controls:
    python -u run_pinn_openbc_suite_v3.py --preset full \
      --cp-data-csv xfoil_cp_dataset_naca2412_dense_Re200k/xfoil_cp_dataset.csv \
      --outdir runs_pinn_openbc_v3_full_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import pandas as pd

from pinn_airfoil_openbc_lab_v3 import (
    build_surface_quadrature,
    ensure_dir,
    get_device,
    make_cases,
    make_suite_summary_plots,
    train_case,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "full", "overnight"], default="full")
    p.add_argument("--outdir", default="runs_pinn_openbc_v3_full")
    p.add_argument("--device", default="auto")
    p.add_argument("--cp-data-csv", default="", help="Optional XFOIL CSV for data-prior controls")
    p.add_argument("--only", default="", help="Comma-separated substrings; run only cases whose name contains one of them")
    p.add_argument("--skip", default="", help="Comma-separated substrings; skip cases whose name contains one of them")
    p.add_argument("--no-save-model", action="store_true", help="Do not save .pt model files")
    p.add_argument("--n-surface", type=int, default=450)
    p.add_argument("--continue-on-error", action="store_true", default=True)
    return p.parse_args()


def split_filters(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Outdir: {outdir}")
    print(f"Preset: {args.preset}")

    surface = build_surface_quadrature(n_per_side=args.n_surface)
    cases = make_cases(args.preset, cp_data_csv=args.cp_data_csv)

    only = split_filters(args.only)
    skip = split_filters(args.skip)
    if only:
        cases = [c for c in cases if any(s in c.name for s in only)]
    if skip:
        cases = [c for c in cases if not any(s in c.name for s in skip)]

    with open(outdir / "suite_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "preset": args.preset,
            "cp_data_csv": args.cp_data_csv,
            "cases": [c.name for c in cases],
            "n_surface": args.n_surface,
        }, f, indent=2)

    print("Cases:")
    for i, c in enumerate(cases, 1):
        print(f"  {i:02d}. {c.name}  Re={c.Re:g}  steps={c.n_steps}  data={c.w_data}  kutta={c.w_kutta}")

    status_rows = []
    for i, case in enumerate(cases, 1):
        print("\n" + "=" * 80)
        print(f"RUN {i}/{len(cases)}: {case.name}")
        print("=" * 80)
        try:
            result = train_case(case, outdir=outdir, device=device, surface=surface, save_model=not args.no_save_model)
            coeffs = result["final"]["coeffs"]
            status_rows.append({
                "case": case.name,
                "status": "ok",
                "CL_pressure": coeffs.get("CL_pressure"),
                "CD_pressure": coeffs.get("CD_pressure"),
                "CM_ref": coeffs.get("CM_ref"),
                "case_dir": str(result["case_dir"]),
            })
        except Exception as e:
            print(f"ERROR in {case.name}: {e}")
            traceback.print_exc()
            status_rows.append({"case": case.name, "status": "error", "error": str(e)})
            if not args.continue_on_error:
                raise
        pd.DataFrame(status_rows).to_csv(outdir / "run_status.csv", index=False)

    print("\nCreating summary plots...")
    summary = make_suite_summary_plots(outdir)
    print(summary)
    print(f"\nDone. Main summary: {outdir / 'summary.csv'}")
    print(f"Plots: {outdir / 'summary_CL_bar.png'}, {outdir / 'summary_re_sweep.png'}")


if __name__ == "__main__":
    main()
