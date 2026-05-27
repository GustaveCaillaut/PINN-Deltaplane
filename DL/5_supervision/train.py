import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# V9 supervised XFOIL surface surrogate
# ============================================================
# Goal:
#   Learn Cp on the airfoil surface from XFOIL data.
#
# Model:
#   input  = [x_pinn, y, side, sin(alpha), cos(alpha)]
#   output = Cp
#
# where:
#   side = +1 for upper surface, -1 for lower surface.
#
# Runtime usage:
#   pressure = 0.5 * rho * U^2 * Cp
#   F_coeff = - integral Cp * n ds
#   F       = 0.5 * rho * U^2 * chord * F_coeff   in 2D per unit span.
#
# Notes:
#   - This is not a PINN anymore. It is a supervised aerodynamic surrogate.
#   - Drag from pressure only is incomplete. Use CL mainly; handle CD separately.
# ============================================================


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def surface_to_side(series):
    values = []
    for v in series.astype(str).str.lower().values:
        if "upper" in v or v in ["u", "top", "extrados"]:
            values.append(1.0)
        elif "lower" in v or v in ["l", "bottom", "intrados"]:
            values.append(-1.0)
        else:
            raise ValueError(f"Unknown surface label: {v}")
    return np.array(values, dtype=np.float32)


# ============================================================
# NACA geometry for diagnostics / integration
# ============================================================

def naca4_camber_y(x, m=0.02, p=0.4):
    x = np.asarray(x)
    yc = np.zeros_like(x, dtype=float)
    if abs(m) == 0:
        return yc
    mask = x < p
    yc[mask] = m / p**2 * (2 * p * x[mask] - x[mask] ** 2)
    yc[~mask] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2)
    return yc


def naca4_points(n=700, m=0.02, p=0.4, t=0.12):
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 0, 1))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    if abs(m) > 0:
        mask = x < p
        yc[mask] = m / p**2 * (2.0 * p * x[mask] - x[mask] ** 2)
        dyc[mask] = 2.0 * m / p**2 * (p - x[mask])
        yc[~mask] = m / (1.0 - p) ** 2 * (
            (1.0 - 2.0 * p) + 2.0 * p * x[~mask] - x[~mask] ** 2
        )
        dyc[~mask] = 2.0 * m / (1.0 - p) ** 2 * (p - x[~mask])

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Polygon convention: upper LE->TE, lower TE->LE.
    X = np.concatenate([xu, xl[::-1]]) - 0.5
    Y = np.concatenate([yu, yl[::-1]])
    return X.astype(np.float32), Y.astype(np.float32)


def polygon_signed_area(x, y):
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * np.sum(x * y2 - x2 * y)


def build_surface_geometry(x, y):
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    dx = x_next - x
    dy = y_next - y
    ds = np.sqrt(dx**2 + dy**2) + 1e-12
    xm = 0.5 * (x + x_next)
    ym = 0.5 * (y + y_next)
    area = polygon_signed_area(x, y)

    if area < 0:
        nx = -dy / ds
        ny = dx / ds
    else:
        nx = dy / ds
        ny = -dx / ds

    half = len(x) // 2
    side = np.concatenate([
        np.ones(half, dtype=np.float32),
        -np.ones(len(x) - half, dtype=np.float32),
    ])

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "xm": xm.astype(np.float32),
        "ym": ym.astype(np.float32),
        "nx": nx.astype(np.float32),
        "ny": ny.astype(np.float32),
        "ds": ds.astype(np.float32),
        "side": side.astype(np.float32),
        "half": half,
        "area": float(area),
        "perimeter": float(ds.sum()),
    }


# ============================================================
# Dataset
# ============================================================

def load_dataset(csv_path, alpha_min=None, alpha_max=None):
    df = pd.read_csv(csv_path)
    required = {"alpha_deg", "x_pinn", "y", "Cp", "surface"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df.copy()
    df = df[np.isfinite(df["alpha_deg"]) & np.isfinite(df["x_pinn"]) & np.isfinite(df["y"]) & np.isfinite(df["Cp"])]

    if alpha_min is not None:
        df = df[df["alpha_deg"] >= alpha_min]
    if alpha_max is not None:
        df = df[df["alpha_deg"] <= alpha_max]

    if len(df) == 0:
        raise ValueError("Dataset is empty after filtering.")

    side = surface_to_side(df["surface"])
    alpha_rad = np.deg2rad(df["alpha_deg"].to_numpy(np.float32)).reshape(-1, 1)
    x = df["x_pinn"].to_numpy(np.float32).reshape(-1, 1)
    y = df["y"].to_numpy(np.float32).reshape(-1, 1)
    cp = df["Cp"].to_numpy(np.float32).reshape(-1, 1)

    features = np.concatenate([
        x,
        y,
        side.reshape(-1, 1),
        np.sin(alpha_rad).astype(np.float32),
        np.cos(alpha_rad).astype(np.float32),
    ], axis=1).astype(np.float32)

    return df.reset_index(drop=True), features, cp.astype(np.float32)


def make_splits(df, features, targets, val_fraction, seed, val_mode="random", val_alphas=""):
    rng = np.random.default_rng(seed)
    n = len(df)

    if val_mode == "alpha":
        vals = parse_float_list(val_alphas)
        if not vals:
            # Default: every 5 degrees if present, but keep at least one alpha for val.
            unique = np.array(sorted(df["alpha_deg"].unique()), dtype=float)
            vals = [float(a) for i, a in enumerate(unique) if i % max(2, int(1 / max(val_fraction, 1e-6))) == 0]
        val_mask = df["alpha_deg"].round(6).isin([round(v, 6) for v in vals]).to_numpy()
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("Alpha split produced empty train or val set.")
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(val_fraction * n))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    return (
        features[train_idx], targets[train_idx], df.iloc[train_idx].reset_index(drop=True),
        features[val_idx], targets[val_idx], df.iloc[val_idx].reset_index(drop=True),
    )


# ============================================================
# Model
# ============================================================

class FourierFeatureBlock(nn.Module):
    def __init__(self, n_freqs=16, scale=1.0, include_raw=True):
        super().__init__()
        self.include_raw = include_raw
        n_each = max(1, n_freqs // 2)
        max_freq = max(1.0, float(scale) * n_each)
        freqs = 2.0 ** torch.linspace(0.0, math.log2(max_freq), n_each)
        # Axis-aligned features for x,y only.
        Bx = torch.stack([freqs, torch.zeros_like(freqs)], dim=0)
        By = torch.stack([torch.zeros_like(freqs), freqs], dim=0)
        B = torch.cat([Bx, By], dim=1)
        if B.shape[1] > n_freqs:
            B = B[:, :n_freqs]
        elif B.shape[1] < n_freqs:
            extra = torch.randn(2, n_freqs - B.shape[1]) * float(scale)
            B = torch.cat([B, extra], dim=1)
        self.register_buffer("B", B.float())
        self.out_dim = 2 * B.shape[1] + (2 if include_raw else 0)

    def forward(self, xy):
        proj = xy @ self.B
        ff = torch.cat([torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], dim=-1)
        if self.include_raw:
            return torch.cat([xy, ff], dim=-1)
        return ff


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CpSurrogate(nn.Module):
    def __init__(self, input_mean, input_std, cp_mean, cp_std, width=128, depth=5, dropout=0.0,
                 use_fourier=True, fourier_freqs=16, fourier_scale=1.0):
        super().__init__()
        self.register_buffer("input_mean", torch.tensor(input_mean.reshape(1, -1), dtype=torch.float32))
        self.register_buffer("input_std", torch.tensor(input_std.reshape(1, -1), dtype=torch.float32))
        self.register_buffer("cp_mean", torch.tensor([[float(cp_mean)]], dtype=torch.float32))
        self.register_buffer("cp_std", torch.tensor([[float(cp_std)]], dtype=torch.float32))
        self.use_fourier = bool(use_fourier)

        if self.use_fourier:
            self.ff = FourierFeatureBlock(n_freqs=fourier_freqs, scale=fourier_scale, include_raw=True)
            in_dim = self.ff.out_dim + 3  # side, sin, cos normalized/raw after input normalization
        else:
            self.ff = None
            in_dim = 5

        layers = [nn.Linear(in_dim, width), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(width, dropout=dropout))
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x = [x_pinn, y, side, sin(alpha), cos(alpha)]
        z = (x - self.input_mean) / self.input_std
        if self.use_fourier:
            xy = z[:, 0:2]
            rest = z[:, 2:5]
            feat = torch.cat([self.ff(xy), rest], dim=1)
        else:
            feat = z
        cp_norm = self.net(feat)
        cp = self.cp_mean + self.cp_std * cp_norm
        return cp


# ============================================================
# Evaluation helpers
# ============================================================

@torch.no_grad()
def predict_np(model, features, device, batch_size=65536):
    model.eval()
    outs = []
    for i in range(0, len(features), batch_size):
        xb = torch.tensor(features[i:i+batch_size], dtype=torch.float32, device=device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def metrics(pred, true):
    err = pred.reshape(-1) - true.reshape(-1)
    return {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def make_surface_features(surface, alpha_deg):
    n = len(surface["xm"])
    a = math.radians(alpha_deg)
    return np.stack([
        surface["xm"],
        surface["ym"],
        surface["side"],
        np.full(n, math.sin(a), dtype=np.float32),
        np.full(n, math.cos(a), dtype=np.float32),
    ], axis=1).astype(np.float32)


def integrate_coefficients(surface, cp, alpha_deg):
    # Coefficients per unit span, chord=1.
    Fx_c = -float(np.sum(cp * surface["nx"] * surface["ds"]))
    Fy_c = -float(np.sum(cp * surface["ny"] * surface["ds"]))
    a = math.radians(alpha_deg)
    eD = np.array([math.cos(a), math.sin(a)])
    eL = np.array([-math.sin(a), math.cos(a)])
    F = np.array([Fx_c, Fy_c])
    CDp = float(F @ eD)
    CLp = float(F @ eL)
    return {"Fx_coeff": Fx_c, "Fy_coeff": Fy_c, "CD_pressure": CDp, "CL_pressure": CLp}


def reference_cp_for_alpha(df, alpha_deg):
    g = df[np.isclose(df["alpha_deg"], alpha_deg)]
    return g


def plot_training_curves(history, outdir):
    hist = pd.DataFrame(history)
    hist.to_csv(outdir / "history.csv", index=False)

    plt.figure(figsize=(9, 5))
    for col in ["train_loss", "val_loss"]:
        if col in hist:
            plt.semilogy(hist["epoch"], hist[col], label=col)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for col in ["train_mae", "val_mae"]:
        if col in hist:
            plt.plot(hist["epoch"], hist[col], label=col)
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mae_curve.png", dpi=150)
    plt.close()


def run_diagnostics(model, df_all, train_df, val_df, surface, device, outdir, diag_alphas, sweep_alphas):
    coef_rows = []
    cp_plot_dir = outdir / "cp_plots"
    cp_plot_dir.mkdir(exist_ok=True)

    for ad in diag_alphas:
        feat = make_surface_features(surface, ad)
        cp_pred = predict_np(model, feat, device).reshape(-1)
        coefs = integrate_coefficients(surface, cp_pred, ad)
        coefs["alpha_deg"] = ad
        coef_rows.append(coefs)

        half = surface["half"]
        xu = surface["xm"][:half]
        xl = surface["xm"][half:][::-1]
        cpu = cp_pred[:half]
        cpl = cp_pred[half:][::-1]

        g = reference_cp_for_alpha(df_all, ad)

        tag = f"alpha_{ad:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        plt.figure(figsize=(9, 4))
        plt.plot(xu, cpu, label="model upper")
        plt.plot(xl, cpl, label="model lower")
        if len(g):
            gu = g[g["surface"].astype(str).str.lower().str.contains("upper")]
            gl = g[g["surface"].astype(str).str.lower().str.contains("lower")]
            plt.scatter(gu["x_pinn"], gu["Cp"], s=8, alpha=0.45, label="XFOIL upper")
            plt.scatter(gl["x_pinn"], gl["Cp"], s=8, alpha=0.45, label="XFOIL lower")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("Cp")
        plt.title(f"Cp at alpha={ad:+.1f} deg, CLp={coefs['CL_pressure']:+.3f}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(cp_plot_dir / f"surface_Cp_{tag}.png", dpi=150)
        plt.close()

        x_common = np.linspace(max(xu.min(), xl.min()), min(xu.max(), xl.max()), 300)
        cpu_i = np.interp(x_common, np.sort(xu), cpu[np.argsort(xu)])
        cpl_i = np.interp(x_common, np.sort(xl), cpl[np.argsort(xl)])
        plt.figure(figsize=(9, 4))
        plt.plot(x_common, cpl_i - cpu_i, label="model lower - upper")
        plt.axhline(0, color="k", linewidth=0.8)
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("Cp jump")
        plt.title(f"Cp jump at alpha={ad:+.1f} deg")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cp_plot_dir / f"surface_Cp_jump_{tag}.png", dpi=150)
        plt.close()

    pd.DataFrame(coef_rows).sort_values("alpha_deg").to_csv(outdir / "coefficients_diag.csv", index=False)

    if sweep_alphas:
        rows = []
        for ad in sweep_alphas:
            feat = make_surface_features(surface, ad)
            cp_pred = predict_np(model, feat, device).reshape(-1)
            coefs = integrate_coefficients(surface, cp_pred, ad)
            coefs["alpha_deg"] = ad
            rows.append(coefs)
        sdf = pd.DataFrame(rows).sort_values("alpha_deg")
        sdf.to_csv(outdir / "alpha_sweep.csv", index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(sdf["alpha_deg"], sdf["CL_pressure"], marker="o", label="model CL pressure")
        plt.axhline(0, color="k", linewidth=0.8)
        plt.grid(True)
        plt.xlabel("alpha [deg]")
        plt.ylabel("CL pressure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "alpha_sweep_CL.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(sdf["alpha_deg"], sdf["CD_pressure"], marker="o", label="model CD pressure")
        plt.axhline(0, color="k", linewidth=0.8)
        plt.grid(True)
        plt.xlabel("alpha [deg]")
        plt.ylabel("CD pressure only")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "alpha_sweep_CDp.png", dpi=150)
        plt.close()


# ============================================================
# Train
# ============================================================

def train(args):
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}", flush=True)
    print(f"Outdir: {outdir}", flush=True)

    df_all, X, y = load_dataset(args.csv, alpha_min=args.alpha_min, alpha_max=args.alpha_max)
    print(f"Loaded dataset: {len(df_all)} rows", flush=True)
    print(f"Alphas: {df_all['alpha_deg'].min()} to {df_all['alpha_deg'].max()}, count={df_all['alpha_deg'].nunique()}", flush=True)

    Xtr, ytr, train_df, Xva, yva, val_df = make_splits(
        df_all, X, y, args.val_fraction, args.seed, val_mode=args.val_mode, val_alphas=args.val_alphas
    )
    print(f"Train rows: {len(Xtr)}, Val rows: {len(Xva)}", flush=True)

    input_mean = Xtr.mean(axis=0)
    input_std = Xtr.std(axis=0) + 1e-8
    cp_mean = float(ytr.mean())
    cp_std = float(ytr.std() + 1e-8)

    model = CpSurrogate(
        input_mean=input_mean,
        input_std=input_std,
        cp_mean=cp_mean,
        cp_std=cp_std,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        use_fourier=not args.no_fourier,
        fourier_freqs=args.fourier_freqs,
        fourier_scale=args.fourier_scale,
    ).to(device)

    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr_min)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_abs = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            train_abs.append(float(torch.mean(torch.abs(pred.detach() - yb)).cpu()))

        scheduler.step()

        model.eval()
        val_losses = []
        val_abs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).cpu()))
                val_abs.append(float(torch.mean(torch.abs(pred - yb)).cpu()))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_mae": float(np.mean(train_abs)),
            "val_loss": float(np.mean(val_losses)),
            "val_mae": float(np.mean(val_abs)),
            "lr": float(opt.param_groups[0]["lr"]),
            "time_sec": float(time.time() - t0),
        }
        history.append(row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_mean": input_mean,
                "input_std": input_std,
                "cp_mean": cp_mean,
                "cp_std": cp_std,
                "args": vars(args),
            }, outdir / "best_model.pt")

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[{epoch:5d}/{args.epochs}] "
                f"train_mse={row['train_loss']:.4e} train_mae={row['train_mae']:.4e} "
                f"val_mse={row['val_loss']:.4e} val_mae={row['val_mae']:.4e} "
                f"best={best_val:.4e}@{best_epoch} lr={row['lr']:.2e} time={row['time_sec']:.1f}s",
                flush=True,
            )

    # Load best model before diagnostics/export.
    ckpt = torch.load(outdir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
        "cp_mean": cp_mean,
        "cp_std": cp_std,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
    }, outdir / "final_model.pt")

    # Numerical dataset metrics.
    pred_train = predict_np(model, Xtr, device)
    pred_val = predict_np(model, Xva, device)
    train_metrics = metrics(pred_train, ytr)
    val_metrics = metrics(pred_val, yva)

    ax, ay = naca4_points(args.naca_n, args.naca_m, args.naca_p, args.naca_t)
    surface = build_surface_geometry(ax, ay)

    plot_training_curves(history, outdir)
    diag_alphas = parse_float_list(args.diag_alphas)
    sweep_alphas = parse_float_list(args.sweep_alphas)
    run_diagnostics(model, df_all, train_df, val_df, surface, device, outdir, diag_alphas, sweep_alphas)

    summary = {
        "csv": str(args.csv),
        "num_rows": int(len(df_all)),
        "num_alphas": int(df_all["alpha_deg"].nunique()),
        "alpha_min": float(df_all["alpha_deg"].min()),
        "alpha_max": float(df_all["alpha_deg"].max()),
        "train_rows": int(len(Xtr)),
        "val_rows": int(len(Xva)),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "input_mean": input_mean,
        "input_std": input_std,
        "cp_mean": cp_mean,
        "cp_std": cp_std,
        "model": {
            "width": args.width,
            "depth": args.depth,
            "dropout": args.dropout,
            "fourier": not args.no_fourier,
            "fourier_freqs": args.fourier_freqs,
            "fourier_scale": args.fourier_scale,
        },
        "inputs": ["x_pinn", "y", "side", "sin_alpha", "cos_alpha"],
        "outputs": ["Cp"],
        "note": "CD_pressure is pressure drag only; do not treat it as total aerodynamic drag.",
    }
    summary = make_json_safe(summary)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))

    # Export ONNX: input shape (N,5), output Cp shape (N,1).
    if args.export_onnx:
        model.eval()
        dummy = torch.zeros(1, 5, dtype=torch.float32, device=device)
        onnx_path = outdir / "cp_surrogate_v9.onnx"
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["features"],
            output_names=["Cp"],
            dynamic_axes={"features": {0: "batch"}, "Cp": {0: "batch"}},
            opset_version=17,
        )
        print(f"Exported ONNX: {onnx_path}", flush=True)

    print("Done.", flush=True)
    print(f"Best val MSE={best_val:.6e}, val MAE={val_metrics['mae']:.6e}", flush=True)


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="XFOIL Cp CSV from generate_xfoil_cp_dataset.py")
    p.add_argument("--outdir", type=str, default="runs_v9_cp_surrogate")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--alpha-min", type=float, default=None)
    p.add_argument("--alpha-max", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--val-mode", type=str, default="random", choices=["random", "alpha"])
    p.add_argument("--val-alphas", type=str, default="")

    p.add_argument("--width", type=int, default=192)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no-fourier", action="store_true")
    p.add_argument("--fourier-freqs", type=int, default=24)
    p.add_argument("--fourier-scale", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--batch-size-eval", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-min", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--print-every", type=int, default=50)

    p.add_argument("--naca-m", type=float, default=0.02)
    p.add_argument("--naca-p", type=float, default=0.4)
    p.add_argument("--naca-t", type=float, default=0.12)
    p.add_argument("--naca-n", type=int, default=900)

    p.add_argument("--diag-alphas", type=str, default="-20,-15,-10,-5,0,5,10,15,20")
    p.add_argument("--sweep-alphas", type=str, default="-25,-20,-15,-10,-5,0,5,10,15,20,25")
    p.add_argument("--export-onnx", action="store_true", default=True)
    p.add_argument("--no-export-onnx", dest="export_onnx", action="store_false")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
