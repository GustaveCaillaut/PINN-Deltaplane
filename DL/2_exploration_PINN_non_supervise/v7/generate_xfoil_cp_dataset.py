import argparse
import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def naca4_camber_y(x, m=0.02, p=0.4):
    x = np.asarray(x)
    yc = np.zeros_like(x, dtype=float)

    mask = x < p
    yc[mask] = m / p**2 * (2 * p * x[mask] - x[mask] ** 2)
    yc[~mask] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2)

    return yc


def parse_alpha_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_xfoil_for_alpha(
    xfoil_cmd,
    naca,
    alpha_deg,
    reynolds,
    mach,
    n_iter,
    outdir,
    viscous=True,
    timeout=60,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tag = f"alpha_{alpha_deg:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    cp_filename = f"cp_{tag}.dat"
    log_filename = f"xfoil_{tag}.log"

    cp_path = outdir / cp_filename
    log_path = outdir / log_filename

    if cp_path.exists():
        cp_path.unlink()

    if viscous:
        viscous_block = f"""
OPER
VISC {reynolds}
MACH {mach}
ITER {n_iter}
ALFA {alpha_deg}
CPWR
{cp_filename}
"""
    else:
        viscous_block = f"""
OPER
MACH {mach}
ITER {n_iter}
ALFA {alpha_deg}
CPWR
{cp_filename}
"""

    commands = f"""
PLOP
G F

NACA {naca}
PANE
{viscous_block}
QUIT
"""

    result = subprocess.run(
        [xfoil_cmd],
        input=commands,
        text=True,
        capture_output=True,
        cwd=str(outdir),
        timeout=timeout,
    )

    log_path.write_text(
        "=== COMMANDS ===\n"
        + commands
        + "\n=== STDOUT ===\n"
        + result.stdout
        + "\n=== STDERR ===\n"
        + result.stderr
        + "\n",
        encoding="utf-8",
    )

    # if result.returncode != 0:
    #     raise RuntimeError(f"XFOIL failed for alpha={alpha_deg}. See {log_path}")

    if not cp_path.exists() or cp_path.stat().st_size == 0:
        raise RuntimeError(
            f"No Cp file produced for alpha={alpha_deg}. "
            f"Return code={result.returncode}. See {log_path}"
            )

    return cp_path, log_path


def parse_cp_file(cp_path, alpha_deg, reynolds, naca, m=0.02, p=0.4):
    rows = []

    with open(cp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            vals = []

            for part in parts:
                try:
                    vals.append(float(part))
                except ValueError:
                    pass

            if len(vals) >= 3:
                x, y, cp = vals[0], vals[1], vals[2]

                # Ignore clearly invalid points.
                if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(cp):
                    continue

                # XFOIL coordinates are chord-based x in [0, 1].
                # Our PINN geometry is centered, so x_pinn = x - 0.5.
                x_clip = min(max(x, 0.0), 1.0)
                yc = naca4_camber_y(np.array([x_clip]), m=m, p=p)[0]
                surface = "upper" if y >= yc else "lower"

                rows.append(
                    {
                        "alpha_deg": alpha_deg,
                        "Re": reynolds,
                        "naca": str(naca),
                        "x_xfoil": x,
                        "x_pinn": x - 0.5,
                        "y": y,
                        "Cp": cp,
                        "surface": surface,
                        "source_file": cp_path.name,
                    }
                )

    if len(rows) == 0:
        raise RuntimeError(f"Could not parse Cp data from {cp_path}")

    return pd.DataFrame(rows)


def plot_cp_by_alpha(df, outdir):
    outdir = Path(outdir)

    for alpha, g in df.groupby("alpha_deg"):
        plt.figure(figsize=(8, 4))

        for surface, gs in g.groupby("surface"):
            gs = gs.sort_values("x_xfoil")
            plt.plot(gs["x_xfoil"], gs["Cp"], label=surface)

        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel("x/c")
        plt.ylabel("Cp")
        plt.title(f"XFOIL Cp, alpha={alpha:+.1f} deg")
        plt.legend()
        plt.tight_layout()

        tag = f"alpha_{alpha:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        plt.savefig(outdir / f"cp_plot_{tag}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    for alpha, g in df.groupby("alpha_deg"):
        # Quick diagnostic: Cp_lower - Cp_upper after interpolation on common x.
        upper = g[g["surface"] == "upper"].sort_values("x_xfoil")
        lower = g[g["surface"] == "lower"].sort_values("x_xfoil")

        if len(upper) < 5 or len(lower) < 5:
            continue

        x_common = np.linspace(0.02, 0.98, 150)
        cp_u = np.interp(x_common, upper["x_xfoil"], upper["Cp"])
        cp_l = np.interp(x_common, lower["x_xfoil"], lower["Cp"])
        jump = cp_l - cp_u

        plt.plot(x_common, jump, label=f"{alpha:+.1f}°")

    plt.grid()
    plt.xlabel("x/c")
    plt.ylabel("Cp_lower - Cp_upper")
    plt.title("Cp jump from XFOIL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "cp_jump_all_alphas.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xfoil-cmd", type=str, default="xfoil")
    parser.add_argument("--naca", type=str, default="2412")
    parser.add_argument("--alphas", type=str, default="-10,-5,0,5,10,15")
    parser.add_argument("--re", type=float, default=2e5)
    parser.add_argument("--mach", type=float, default=0.0)
    parser.add_argument("--iter", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="xfoil_cp_dataset")
    parser.add_argument("--timeout", type=int, default=60)

    # NACA 4-digit parameters used only for upper/lower labeling.
    parser.add_argument("--naca-m", type=float, default=0.02)
    parser.add_argument("--naca-p", type=float, default=0.4)

    parser.add_argument("--inviscid", action="store_true")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    alphas = parse_alpha_list(args.alphas)
    all_dfs = []

    print(f"Using XFOIL command: {args.xfoil_cmd}")
    print(f"NACA {args.naca}")
    print(f"alphas = {alphas}")
    print(f"Re = {args.re}")
    print(f"viscous = {not args.inviscid}")
    print(f"outdir = {outdir}")

    for alpha in alphas:
        print(f"\n=== Running alpha={alpha:+.2f} deg ===")

        cp_path, log_path = run_xfoil_for_alpha(
            xfoil_cmd=args.xfoil_cmd,
            naca=args.naca,
            alpha_deg=alpha,
            reynolds=args.re,
            mach=args.mach,
            n_iter=args.iter,
            outdir=outdir,
            viscous=not args.inviscid,
            timeout=args.timeout,
        )

        df = parse_cp_file(
            cp_path=cp_path,
            alpha_deg=alpha,
            reynolds=args.re,
            naca=args.naca,
            m=args.naca_m,
            p=args.naca_p,
        )

        print(f"Parsed {len(df)} Cp points from {cp_path.name}")
        print(f"  Cp min/max: {df['Cp'].min():+.4f}, {df['Cp'].max():+.4f}")
        print(f"  upper points: {(df['surface'] == 'upper').sum()}")
        print(f"  lower points: {(df['surface'] == 'lower').sum()}")

        all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)

    csv_path = outdir / "xfoil_cp_dataset.csv"
    full.to_csv(csv_path, index=False)

    print(f"\nSaved dataset: {csv_path}")

    plot_cp_by_alpha(full, outdir)

    print(f"Saved plots in: {outdir}")
    print("Done.")


if __name__ == "__main__":
    main()