import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.path import Path as MplPath


# ============================================================
# V8: alpha-parametric PINN + XFOIL Cp data + corrected Kutta
# ============================================================
# Main goal:
#   Train a real alpha-dependent model
#       f(x, y, alpha) -> (u, v, Cp)
#   with:
#       - Navier-Stokes residual
#       - no-slip wall
#       - inclined inflow boundary
#       - pressure gauge at one point
#       - optional sparse XFOIL Cp surface data
#       - corrected Kutta-style wake alignment
#
# Important correction vs V7 Kutta:
#   We do NOT impose Cp_upper = Cp_lower over a TE neighborhood.
#   We do NOT impose full velocity equality upper/lower.
#   Instead, we impose a local wake/tangency condition behind the trailing edge:
#       velocity normal to wake direction should be small.
#   This is weaker and less anti-lift.
#
# Coordinates:
#   NACA chord c=1 centered at x=0.
#   LE = -0.5, TE = +0.5.
#
# Model input:
#   x, y, alpha_rad.
#   Internally uses sin(alpha), cos(alpha).
#
# Output:
#   u, v, Cp.
#   p = 0.5 * U^2 * Cp in the PDE residual.
# ============================================================


# ============================================================
# Utilities
# ============================================================

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


def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Airfoil geometry
# ============================================================

def naca4_camber_y(x, m=0.02, p=0.4):
    x = np.asarray(x)
    yc = np.zeros_like(x, dtype=float)
    if abs(m) == 0:
        return yc
    mask = x < p
    yc[mask] = m / p**2 * (2.0 * p * x[mask] - x[mask] ** 2)
    yc[~mask] = m / (1.0 - p) ** 2 * (
        (1.0 - 2.0 * p) + 2.0 * p * x[~mask] - x[~mask] ** 2
    )
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

    # Closed polygon convention used by previous scripts:
    # upper: LE -> TE, lower: TE -> LE.
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

    # If polygon is clockwise, outward is left normal. Else right normal.
    if area < 0:
        nx = -dy / ds
        ny = dx / ds
    else:
        nx = dy / ds
        ny = -dx / ds

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "xm": xm.astype(np.float32),
        "ym": ym.astype(np.float32),
        "nx": nx.astype(np.float32),
        "ny": ny.astype(np.float32),
        "ds": ds.astype(np.float32),
        "area": float(area),
        "perimeter": float(ds.sum()),
        "half": len(x) // 2,
    }


# ============================================================
# Optional XFOIL dataset generation
# ============================================================

def maybe_generate_xfoil(args):
    if not args.make_xfoil_data:
        return
    outdir = Path(args.xfoil_outdir)
    csv_path = outdir / "xfoil_cp_dataset.csv"
    if csv_path.exists() and not args.force_xfoil_data:
        print(f"XFOIL dataset already exists: {csv_path}")
        return
    script = Path(args.xfoil_generator)
    if not script.exists():
        raise FileNotFoundError(f"Cannot find generator script: {script}")

    cmd = [
        "python",
        str(script),
        f"--alphas={args.xfoil_alphas}",
        f"--re={args.xfoil_re}",
        f"--iter={args.xfoil_iter}",
        f"--outdir={outdir}",
        f"--naca-m={args.naca_m}",
        f"--naca-p={args.naca_p}",
        f"--naca-t={args.naca_t}",
    ]
    if args.xfoil_inviscid:
        cmd.append("--inviscid")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ============================================================
# Model
# ============================================================

class FourierEncoder(nn.Module):
    def __init__(self, mode="multiscale", n_freqs=16, scale=1.0, include_raw=True):
        super().__init__()
        self.mode = mode
        self.include_raw = include_raw
        self.n_freqs = int(n_freqs)
        self.scale = float(scale)

        if mode == "none":
            self.register_buffer("B", torch.empty(2, 0))
            self.out_dim = 2
            return

        if mode == "random":
            B = torch.randn(2, self.n_freqs) * self.scale
        elif mode == "multiscale":
            n_each = max(1, self.n_freqs // 2)
            max_freq = max(1.0, self.scale * n_each)
            freqs = 2.0 ** torch.linspace(0.0, math.log2(max_freq), n_each)
            Bx = torch.stack([freqs, torch.zeros_like(freqs)], dim=0)
            By = torch.stack([torch.zeros_like(freqs), freqs], dim=0)
            B = torch.cat([Bx, By], dim=1)
            if B.shape[1] > self.n_freqs:
                B = B[:, : self.n_freqs]
            elif B.shape[1] < self.n_freqs:
                extra = torch.randn(2, self.n_freqs - B.shape[1]) * self.scale
                B = torch.cat([B, extra], dim=1)
        else:
            raise ValueError(mode)

        self.register_buffer("B", B.float())
        self.out_dim = 2 * B.shape[1] + (2 if include_raw else 0)

    def forward(self, xy_norm):
        if self.mode == "none":
            return xy_norm
        proj = xy_norm @ self.B
        ff = torch.cat([
            torch.sin(2.0 * math.pi * proj),
            torch.cos(2.0 * math.pi * proj),
        ], dim=-1)
        if self.include_raw:
            return torch.cat([xy_norm, ff], dim=-1)
        return ff


class AlphaAirfoilPINN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.register_buffer("xy_center", torch.tensor([
            0.5 * (args.x_min + args.x_max),
            0.5 * (args.y_min + args.y_max),
        ], dtype=torch.float32))
        self.register_buffer("xy_scale", torch.tensor([
            0.5 * (args.x_max - args.x_min),
            0.5 * (args.y_max - args.y_min),
        ], dtype=torch.float32))

        self.encoder = FourierEncoder(
            mode=args.fourier_mode,
            n_freqs=args.fourier_n_freqs,
            scale=args.fourier_scale,
            include_raw=args.fourier_include_raw,
        )

        # Features = encoded x,y + sin(alpha), cos(alpha).
        in_dim = self.encoder.out_dim + 2
        layers = [nn.Linear(in_dim, args.width), nn.Tanh()]
        for _ in range(args.depth - 1):
            layers += [nn.Linear(args.width, args.width), nn.Tanh()]
        layers += [nn.Linear(args.width, 3)]
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, xya):
        xy = xya[:, 0:2]
        alpha = xya[:, 2:3]
        xy_norm = (xy - self.xy_center) / self.xy_scale
        xy_feat = self.encoder(xy_norm)
        a_feat = torch.cat([torch.sin(alpha), torch.cos(alpha)], dim=1)
        return self.net(torch.cat([xy_feat, a_feat], dim=1))


# ============================================================
# Trainer
# ============================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.U = float(args.U)
        self.q = 0.5 * self.U**2
        self.nu = float(args.nu)
        self.alpha_fixed = math.radians(args.alpha_deg)
        self.alpha_min = math.radians(args.alpha_min_deg)
        self.alpha_max = math.radians(args.alpha_max_deg)

        self.ax, self.ay = naca4_points(args.naca_n, args.naca_m, args.naca_p, args.naca_t)
        self.airfoil_path = MplPath(np.stack([self.ax, self.ay], axis=1))
        self.surf = build_surface_geometry(self.ax, self.ay)
        self.normals_ok, self.frac_plus_out, self.frac_minus_in = self.check_normals()

        self.surf_xy_np = np.stack([self.surf["xm"], self.surf["ym"]], axis=1).astype(np.float32)
        self.surf_nx_t = torch.tensor(self.surf["nx"].reshape(-1, 1), device=self.device)
        self.surf_ny_t = torch.tensor(self.surf["ny"].reshape(-1, 1), device=self.device)
        self.surf_ds_t = torch.tensor(self.surf["ds"].reshape(-1, 1), device=self.device)

        self.model = AlphaAirfoilPINN(args).to(self.device)
        self.history = []

        self.cp_df = None
        self.cp_arrays = None
        if args.use_data:
            self.load_cp_data()

    # ------------------------ setup ------------------------
    def check_normals(self, eps=1e-3):
        pts = np.stack([self.surf["xm"], self.surf["ym"]], axis=1)
        n = np.stack([self.surf["nx"], self.surf["ny"]], axis=1)
        plus = pts + eps * n
        minus = pts - eps * n
        plus_inside = self.airfoil_path.contains_points(plus)
        minus_inside = self.airfoil_path.contains_points(minus)
        frac_plus_out = float(np.mean(~plus_inside))
        frac_minus_in = float(np.mean(minus_inside))
        return frac_plus_out > 0.95 and frac_minus_in > 0.95, frac_plus_out, frac_minus_in

    def load_cp_data(self):
        path = Path(self.args.cp_data_csv)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"alpha_deg", "x_pinn", "y", "Cp", "surface"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Cp data missing columns: {missing}")

        if self.args.data_alpha_filter:
            allowed = parse_float_list(self.args.data_alpha_filter)
            df = df[df["alpha_deg"].round(6).isin([round(a, 6) for a in allowed])]

        df = df.copy()
        df = df[np.isfinite(df["x_pinn"]) & np.isfinite(df["y"]) & np.isfinite(df["Cp"])]
        if len(df) == 0:
            raise ValueError("Empty Cp dataset after filtering.")

        # Optional leading-edge boost: pressure peak is often there.
        x01 = np.clip(df["x_pinn"].to_numpy() + 0.5, 0.0, 1.0)
        weights = np.ones(len(df), dtype=np.float64)
        if self.args.data_le_boost > 0:
            weights += self.args.data_le_boost / np.sqrt(x01 + 0.02)
        weights /= weights.sum()

        self.cp_df = df.reset_index(drop=True)
        self.cp_arrays = {
            "x": self.cp_df["x_pinn"].to_numpy(np.float32).reshape(-1, 1),
            "y": self.cp_df["y"].to_numpy(np.float32).reshape(-1, 1),
            "alpha": np.deg2rad(self.cp_df["alpha_deg"].to_numpy(np.float32).reshape(-1, 1)),
            "cp": self.cp_df["Cp"].to_numpy(np.float32).reshape(-1, 1),
            "weights": weights,
        }
        print(f"Loaded Cp data: {len(self.cp_df)} rows")
        print("Data alphas:", sorted(self.cp_df["alpha_deg"].unique()))

    # ------------------------ sampling ------------------------
    def sample_alpha(self, n):
        if self.args.alpha_sampling == "fixed":
            return np.full((n, 1), self.alpha_fixed, dtype=np.float32)
        return np.random.uniform(self.alpha_min, self.alpha_max, size=(n, 1)).astype(np.float32)

    def make_xya(self, x, y, a, requires_grad=False):
        arr = np.concatenate([x, y, a], axis=1).astype(np.float32)
        t = torch.tensor(arr, device=self.device, dtype=torch.float32)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def sample_fluid_points(self, n):
        pts = []
        count = 0
        while count < n:
            m = int((n - count) * 1.4) + 300
            x = np.random.uniform(self.args.x_min, self.args.x_max, size=(m, 1)).astype(np.float32)
            y = np.random.uniform(self.args.y_min, self.args.y_max, size=(m, 1)).astype(np.float32)
            cand = np.concatenate([x, y], axis=1)
            inside = self.airfoil_path.contains_points(cand)
            fluid = cand[~inside]
            pts.append(fluid)
            count += fluid.shape[0]
        pts = np.concatenate(pts, axis=0)[:n]
        return pts[:, 0:1], pts[:, 1:2]

    def sample_near_wall_points(self, n, r_min=0.002, r_max=0.20):
        idx = np.random.randint(0, len(self.surf["xm"]), size=n)
        x0 = self.surf["xm"][idx].reshape(-1, 1)
        y0 = self.surf["ym"][idx].reshape(-1, 1)
        nx = self.surf["nx"][idx].reshape(-1, 1)
        ny = self.surf["ny"][idx].reshape(-1, 1)
        z = np.random.rand(n, 1).astype(np.float32)
        r = r_min * (r_max / r_min) ** z
        x = x0 + r * nx
        y = y0 + r * ny
        pts = np.concatenate([x, y], axis=1)
        inside = self.airfoil_path.contains_points(pts)
        if inside.any():
            xr, yr = self.sample_fluid_points(int(inside.sum()))
            x[inside] = xr
            y[inside] = yr
        return x.astype(np.float32), y.astype(np.float32)

    def sample_wall(self, n):
        ds = self.surf["ds"]
        prob = ds / ds.sum()
        idx = np.random.choice(len(ds), size=n, replace=True, p=prob)
        x = self.surf["xm"][idx].reshape(-1, 1)
        y = self.surf["ym"][idx].reshape(-1, 1)
        return x.astype(np.float32), y.astype(np.float32)

    def sample_boundary(self, n):
        side = np.random.randint(0, 4, size=n)
        x = np.empty((n, 1), dtype=np.float32)
        y = np.empty((n, 1), dtype=np.float32)
        nx = np.empty((n, 1), dtype=np.float32)
        ny = np.empty((n, 1), dtype=np.float32)
        for s in range(4):
            mask = side == s
            k = int(mask.sum())
            if k == 0:
                continue
            if s == 0:
                x[mask] = self.args.x_min
                y[mask] = np.random.uniform(self.args.y_min, self.args.y_max, size=(k, 1))
                nx[mask] = -1.0
                ny[mask] = 0.0
            elif s == 1:
                x[mask] = self.args.x_max
                y[mask] = np.random.uniform(self.args.y_min, self.args.y_max, size=(k, 1))
                nx[mask] = 1.0
                ny[mask] = 0.0
            elif s == 2:
                x[mask] = np.random.uniform(self.args.x_min, self.args.x_max, size=(k, 1))
                y[mask] = self.args.y_min
                nx[mask] = 0.0
                ny[mask] = -1.0
            else:
                x[mask] = np.random.uniform(self.args.x_min, self.args.x_max, size=(k, 1))
                y[mask] = self.args.y_max
                nx[mask] = 0.0
                ny[mask] = 1.0
        return x, y, nx, ny

    # ------------------------ physics ------------------------
    def grad(self, outputs, inputs):
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    def ns_residual(self, xya):
        out = self.model(xya)
        u = out[:, 0:1]
        v = out[:, 1:2]
        Cp = out[:, 2:3]
        p = self.q * Cp

        gu = self.grad(u, xya)
        gv = self.grad(v, xya)
        gp = self.grad(p, xya)

        ux = gu[:, 0:1]
        uy = gu[:, 1:2]
        vx = gv[:, 0:1]
        vy = gv[:, 1:2]
        px = gp[:, 0:1]
        py = gp[:, 1:2]

        gux = self.grad(ux, xya)
        guy = self.grad(uy, xya)
        gvx = self.grad(vx, xya)
        gvy = self.grad(vy, xya)

        uxx = gux[:, 0:1]
        uyy = guy[:, 1:2]
        vxx = gvx[:, 0:1]
        vyy = gvy[:, 1:2]

        mom_x = u * ux + v * uy + px - self.nu * (uxx + uyy)
        mom_y = u * vx + v * vy + py - self.nu * (vxx + vyy)
        div = ux + vy
        return mom_x, mom_y, div

    def data_loss(self):
        if (not self.args.use_data) or self.cp_arrays is None or self.args.w_data <= 0:
            z = torch.tensor(0.0, device=self.device)
            return z, z
        n = min(self.args.n_data, len(self.cp_arrays["x"]))
        idx = np.random.choice(len(self.cp_arrays["x"]), size=n, replace=True, p=self.cp_arrays["weights"])
        x = self.cp_arrays["x"][idx]
        y = self.cp_arrays["y"][idx]
        a = self.cp_arrays["alpha"][idx]
        cp_ref = torch.tensor(self.cp_arrays["cp"][idx], device=self.device)
        pred = self.model(self.make_xya(x, y, a))[:, 2:3]
        mse = torch.mean((pred - cp_ref) ** 2)
        mae = torch.mean(torch.abs(pred - cp_ref))
        return mse, mae

    def corrected_kutta_loss(self):
        if self.args.w_kutta <= 0:
            z = torch.tensor(0.0, device=self.device)
            return z, z, z

        # Wake points behind TE along freestream direction.
        n = self.args.n_kutta
        a = self.sample_alpha(n)
        ex = np.cos(a).astype(np.float32)
        ey = np.sin(a).astype(np.float32)
        nx = (-np.sin(a)).astype(np.float32)
        ny = (np.cos(a)).astype(np.float32)

        s = np.random.uniform(self.args.wake_s_min, self.args.wake_s_max, size=(n, 1)).astype(np.float32)
        jitter = np.random.normal(0.0, self.args.wake_jitter, size=(n, 1)).astype(np.float32)

        x = 0.5 + s * ex + jitter * nx
        y = 0.0 + s * ey + jitter * ny
        out = self.model(self.make_xya(x, y, a))
        u = out[:, 0:1]
        v = out[:, 1:2]
        nx_t = torch.tensor(nx, device=self.device)
        ny_t = torch.tensor(ny, device=self.device)
        ex_t = torch.tensor(ex, device=self.device)
        ey_t = torch.tensor(ey, device=self.device)

        # Main corrected Kutta condition: velocity tangent to wake.
        v_normal = u * nx_t + v * ny_t
        loss_tangent = torch.mean(v_normal**2)

        # Optional weak recovery further downstream. This is not Kutta itself;
        # it just prevents pathological wakes if enabled.
        if self.args.w_wake_recovery > 0:
            u_tan = u * ex_t + v * ey_t
            loss_recovery = torch.mean((u_tan - self.U) ** 2)
        else:
            loss_recovery = torch.tensor(0.0, device=self.device)

        total = self.args.w_kutta_tangent * loss_tangent + self.args.w_wake_recovery * loss_recovery
        return total, loss_tangent, loss_recovery

    def compute_loss(self, cf=1.0, small=False):
        scale = 0.5 if small else 1.0
        n_f = max(1, int(self.args.n_f * scale))
        n_near = max(1, int(self.args.n_near * scale))
        n_wall = max(1, int(self.args.n_wall * scale))
        n_b = max(1, int(self.args.n_boundary * scale))

        # PDE global.
        x, y = self.sample_fluid_points(n_f)
        a = self.sample_alpha(n_f)
        xya = self.make_xya(x, y, a, requires_grad=True)
        mx, my, div = self.ns_residual(xya)
        loss_pde = torch.mean(mx**2 + my**2)
        loss_div = torch.mean(div**2)

        # PDE near wall.
        x, y = self.sample_near_wall_points(n_near)
        a = self.sample_alpha(n_near)
        xya = self.make_xya(x, y, a, requires_grad=True)
        mxn, myn, divn = self.ns_residual(xya)
        loss_near_pde = torch.mean(mxn**2 + myn**2)
        loss_near_div = torch.mean(divn**2)

        # Boundary inflow/outflow.
        xb, yb, nbx, nby = self.sample_boundary(n_b)
        ab = self.sample_alpha(n_b)
        ux_inf = self.U * np.cos(ab)
        uy_inf = self.U * np.sin(ab)
        dot = ux_inf * nbx + uy_inf * nby
        inflow = dot.reshape(-1) < 0
        outb = self.model(self.make_xya(xb, yb, ab))
        if inflow.any():
            ux_t = torch.tensor(ux_inf[inflow], device=self.device)
            uy_t = torch.tensor(uy_inf[inflow], device=self.device)
            loss_inflow = torch.mean((outb[inflow, 0:1] - ux_t) ** 2 + (outb[inflow, 1:2] - uy_t) ** 2)
        else:
            loss_inflow = torch.tensor(0.0, device=self.device)
        if (~inflow).any():
            loss_outflow_cp = torch.mean(outb[~inflow, 2:3] ** 2)
        else:
            loss_outflow_cp = torch.tensor(0.0, device=self.device)
        loss_far_cp = torch.mean(outb[:, 2:3] ** 2)

        # Wall no-slip.
        xw, yw = self.sample_wall(n_wall)
        aw = self.sample_alpha(n_wall)
        outw = self.model(self.make_xya(xw, yw, aw))
        loss_wall = torch.mean(outw[:, 0:1] ** 2 + outw[:, 1:2] ** 2)

        # Pressure gauge at a single point for sampled alpha.
        ag = self.sample_alpha(1)
        xg = np.array([[self.args.x_pressure_ref if self.args.x_pressure_ref is not None else self.args.x_max]], dtype=np.float32)
        yg = np.array([[self.args.y_pressure_ref]], dtype=np.float32)
        loss_gauge = torch.mean(self.model(self.make_xya(xg, yg, ag))[:, 2:3] ** 2)

        # Corrected Kutta and data.
        loss_kutta, loss_kutta_tan, loss_wake_rec = self.corrected_kutta_loss()
        loss_data, loss_data_mae = self.data_loss()

        loss = (
            cf * self.args.w_pde * loss_pde
            + cf * self.args.w_div * loss_div
            + cf * self.args.w_near_pde * loss_near_pde
            + cf * self.args.w_near_div * loss_near_div
            + self.args.w_inflow_vel * loss_inflow
            + self.args.w_wall * loss_wall
            + self.args.w_pressure_gauge * loss_gauge
            + self.args.w_outflow_cp * loss_outflow_cp
            + self.args.w_far_cp * loss_far_cp
            + cf * self.args.w_kutta * loss_kutta
            + self.args.w_data * loss_data
        )

        # CL diagnostic at alpha_diag only; not used in loss.
        CL_diag = self.compute_coefficients(self.args.alpha_deg, torch_mode=True)["CL"]

        parts = {
            "total": loss,
            "pde": loss_pde,
            "div": loss_div,
            "near_pde": loss_near_pde,
            "near_div": loss_near_div,
            "inflow": loss_inflow,
            "wall": loss_wall,
            "gauge": loss_gauge,
            "outflow_cp": loss_outflow_cp,
            "far_cp": loss_far_cp,
            "kutta": loss_kutta,
            "kutta_tangent": loss_kutta_tan,
            "wake_recovery": loss_wake_rec,
            "data_mse": loss_data,
            "data_mae": loss_data_mae,
            "CL_diag": CL_diag.detach(),
            "cf": torch.tensor(cf, device=self.device),
        }
        return loss, parts

    # ------------------------ diagnostics ------------------------
    def surface_input(self, alpha_deg):
        n = len(self.surf_xy_np)
        a = np.full((n, 1), math.radians(alpha_deg), dtype=np.float32)
        return self.make_xya(self.surf_xy_np[:, 0:1], self.surf_xy_np[:, 1:2], a)

    def compute_coefficients(self, alpha_deg, torch_mode=False):
        xya = self.surface_input(alpha_deg)
        out = self.model(xya)
        Cp = out[:, 2:3]
        p = self.q * Cp
        Fx = -torch.sum(p * self.surf_nx_t * self.surf_ds_t)
        Fy = -torch.sum(p * self.surf_ny_t * self.surf_ds_t)
        a = math.radians(alpha_deg)
        eD = torch.tensor([[math.cos(a)], [math.sin(a)]], device=self.device)
        eL = torch.tensor([[-math.sin(a)], [math.cos(a)]], device=self.device)
        F = torch.stack([Fx, Fy]).reshape(1, 2)
        D = (F @ eD).squeeze()
        L = (F @ eL).squeeze()
        CD = D / self.q
        CL = L / self.q
        if torch_mode:
            return {"Fx": Fx, "Fy": Fy, "D": D, "L": L, "CD": CD, "CL": CL}
        return {k: float(v.detach().cpu()) for k, v in {"Fx": Fx, "Fy": Fy, "D": D, "L": L, "CD": CD, "CL": CL}.items()}

    @torch.no_grad()
    def surface_cp_np(self, alpha_deg):
        return self.model(self.surface_input(alpha_deg))[:, 2].detach().cpu().numpy()

    def config(self):
        return {
            "U": self.U,
            "nu": self.nu,
            "alpha_min_deg": self.args.alpha_min_deg,
            "alpha_max_deg": self.args.alpha_max_deg,
            "alpha_sampling": self.args.alpha_sampling,
            "domain": [self.args.x_min, self.args.x_max, self.args.y_min, self.args.y_max],
            "naca": {"m": self.args.naca_m, "p": self.args.naca_p, "t": self.args.naca_t},
            "normal_check": {"ok": self.normals_ok, "plus_outside": self.frac_plus_out, "minus_inside": self.frac_minus_in},
            "fourier": {"mode": self.args.fourier_mode, "n_freqs": self.args.fourier_n_freqs, "scale": self.args.fourier_scale},
            "loss_weights": {k: getattr(self.args, k) for k in vars(self.args) if k.startswith("w_")},
            "use_data": self.args.use_data,
            "cp_data_csv": self.args.cp_data_csv,
        }

    def save_checkpoint(self, name):
        path = self.outdir / name
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config(), "history": self.history}, path)
        print(f"Saved {path}")

    def run_diagnostics(self):
        self.model.eval()

        if self.history:
            plt.figure(figsize=(11, 6))
            keys = ["total", "pde", "near_pde", "div", "near_div", "wall", "inflow", "kutta", "data_mse"]
            for k in keys:
                vals = np.array([h[k] for h in self.history], dtype=float)
                plt.semilogy(vals + 1e-30, label=k)
            plt.grid(True)
            plt.legend()
            plt.xlabel("logged step")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(self.outdir / "losses.png", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 4))
            vals = np.array([h["CL_diag"] for h in self.history], dtype=float)
            plt.plot(vals, label=f"CL diag alpha={self.args.alpha_deg}")
            plt.axhline(0, color="k", linewidth=0.8)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / "cl_training_diag.png", dpi=150)
            plt.close()

        diag_alphas = parse_float_list(self.args.diag_alphas) or [self.args.alpha_deg]
        coef_rows = []
        for ad in diag_alphas:
            coef = self.compute_coefficients(ad)
            coef["alpha_deg"] = ad
            coef["U"] = self.U
            coef_rows.append(coef)

            Cp = self.surface_cp_np(ad)
            half = self.surf["half"]
            xu = self.surf["xm"][:half]
            xl = self.surf["xm"][half:][::-1]
            cpu = Cp[:half]
            cpl = Cp[half:][::-1]

            tag = f"alpha_{ad:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            plt.figure(figsize=(9, 4))
            plt.plot(xu, cpu, label="PINN upper")
            plt.plot(xl, cpl, label="PINN lower")

            if self.cp_df is not None:
                g = self.cp_df[np.isclose(self.cp_df["alpha_deg"], ad)]
                if len(g):
                    gu = g[g["surface"] == "upper"]
                    gl = g[g["surface"] == "lower"]
                    plt.scatter(gu["x_pinn"], gu["Cp"], s=8, alpha=0.45, label="XFOIL upper")
                    plt.scatter(gl["x_pinn"], gl["Cp"], s=8, alpha=0.45, label="XFOIL lower")

            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("Cp")
            plt.title(f"Cp alpha={ad:+.1f}, CL={coef['CL']:+.4f}")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(self.outdir / f"surface_Cp_{tag}.png", dpi=150)
            plt.close()

            x_common = np.linspace(max(xu.min(), xl.min()), min(xu.max(), xl.max()), 250)
            cpu_i = np.interp(x_common, np.sort(xu), cpu[np.argsort(xu)])
            cpl_i = np.interp(x_common, np.sort(xl), cpl[np.argsort(xl)])
            plt.figure(figsize=(9, 4))
            plt.plot(x_common, cpl_i - cpu_i, label="PINN lower-upper")
            plt.axhline(0, color="k", linewidth=0.8)
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("Cp jump")
            plt.title(f"Cp jump alpha={ad:+.1f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / f"surface_Cp_jump_{tag}.png", dpi=150)
            plt.close()

        coef_df = pd.DataFrame(coef_rows).sort_values("alpha_deg")
        coef_df.to_csv(self.outdir / "coefficients_diag.csv", index=False)

        sweep_alphas = parse_float_list(self.args.sweep_alphas)
        if sweep_alphas:
            rows = []
            for ad in sweep_alphas:
                c = self.compute_coefficients(ad)
                c["alpha_deg"] = ad
                rows.append(c)
            df = pd.DataFrame(rows).sort_values("alpha_deg")
            df.to_csv(self.outdir / "alpha_sweep.csv", index=False)
            plt.figure(figsize=(8, 4))
            plt.plot(df["alpha_deg"], df["CL"], marker="o", label="PINN CL")
            plt.axhline(0, color="k", linewidth=0.8)
            plt.grid(True)
            plt.xlabel("alpha [deg]")
            plt.ylabel("CL")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / "alpha_sweep_CL.png", dpi=150)
            plt.close()

        summary = make_json_safe({
            "config": self.config(),
            "coefficients_diag": coef_rows,
            "history_last": self.history[-1] if self.history else {},
        })
        with open(self.outdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(self.outdir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2))

    def train(self):
        print(f"Device: {self.device}")
        print(f"Outdir: {self.outdir}")
        print(f"Normal check: ok={self.normals_ok}, plus_out={self.frac_plus_out:.3f}, minus_in={self.frac_minus_in:.3f}")
        print(f"Data: {self.args.use_data}, w_data={self.args.w_data}")
        print(f"Corrected Kutta: w_kutta={self.args.w_kutta}, w_tangent={self.args.w_kutta_tangent}")

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        t0 = time.time()
        for step in range(1, self.args.adam_steps + 1):
            opt.zero_grad(set_to_none=True)
            cf = min(1.0, step / max(1, self.args.warmup_steps))
            loss, parts = self.compute_loss(cf=cf, small=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            opt.step()

            if step == 1 or step % self.args.print_every == 0:
                row = {k: float(v.detach().cpu()) for k, v in parts.items()}
                self.history.append(row)
                print(
                    f"[{step:7d}/{self.args.adam_steps}] "
                    f"loss={row['total']:.3e} pde={row['pde']:.2e} near={row['near_pde']:.2e} "
                    f"data={row['data_mse']:.2e} kutta={row['kutta']:.2e} "
                    f"wall={row['wall']:.2e} CLdiag={row['CL_diag']:+.3e} "
                    f"cf={row['cf']:.2f} time={time.time()-t0:.1f}s"
                )

            if self.args.save_every > 0 and step % self.args.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{step}.pt")

        self.save_checkpoint("model_adam.pt")

        if self.args.lbfgs_steps > 0:
            print("Starting LBFGS...")
            opt2 = torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.args.lbfgs_lr,
                max_iter=20,
                max_eval=25,
                history_size=50,
                tolerance_grad=1e-8,
                tolerance_change=1e-10,
                line_search_fn="strong_wolfe",
            )
            for k in range(1, self.args.lbfgs_steps + 1):
                def closure():
                    opt2.zero_grad(set_to_none=True)
                    l, _ = self.compute_loss(cf=1.0, small=True)
                    l.backward()
                    return l
                opt2.step(closure)
                if k == 1 or k % self.args.lbfgs_print_every == 0:
                    _, parts = self.compute_loss(cf=1.0, small=True)
                    row = {kk: float(v.detach().cpu()) for kk, v in parts.items()}
                    self.history.append(row)
                    print(
                        f"[LBFGS {k:5d}/{self.args.lbfgs_steps}] "
                        f"loss={row['total']:.3e} data={row['data_mse']:.2e} "
                        f"kutta={row['kutta']:.2e} CLdiag={row['CL_diag']:+.3e}"
                    )

        self.save_checkpoint("model_final.pt")
        self.run_diagnostics()
        print("Done.")


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="runs_v8_alpha_data_kutta")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1234)

    # Flow / alpha parameterization.
    p.add_argument("--U", type=float, default=2.0)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--alpha-deg", type=float, default=5.0, help="diagnostic alpha")
    p.add_argument("--alpha-min-deg", type=float, default=-15.0)
    p.add_argument("--alpha-max-deg", type=float, default=15.0)
    p.add_argument("--alpha-sampling", type=str, default="range", choices=["range", "fixed"])

    # Domain.
    p.add_argument("--x-min", type=float, default=-4.0)
    p.add_argument("--x-max", type=float, default=6.0)
    p.add_argument("--y-min", type=float, default=-3.0)
    p.add_argument("--y-max", type=float, default=3.0)
    p.add_argument("--x-pressure-ref", type=float, default=None)
    p.add_argument("--y-pressure-ref", type=float, default=0.0)

    # Airfoil.
    p.add_argument("--naca-m", type=float, default=0.02)
    p.add_argument("--naca-p", type=float, default=0.4)
    p.add_argument("--naca-t", type=float, default=0.12)
    p.add_argument("--naca-n", type=int, default=700)

    # Model.
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--fourier-mode", type=str, default="multiscale", choices=["none", "random", "multiscale"])
    p.add_argument("--fourier-n-freqs", type=int, default=16)
    p.add_argument("--fourier-scale", type=float, default=1.0)
    p.add_argument("--fourier-include-raw", action="store_true", default=True)
    p.add_argument("--no-fourier-include-raw", dest="fourier_include_raw", action="store_false")

    # Sampling.
    p.add_argument("--n-f", type=int, default=4096)
    p.add_argument("--n-near", type=int, default=4096)
    p.add_argument("--n-boundary", type=int, default=2048)
    p.add_argument("--n-wall", type=int, default=1024)
    p.add_argument("--n-data", type=int, default=4096)
    p.add_argument("--n-kutta", type=int, default=512)

    # Training.
    p.add_argument("--adam-steps", type=int, default=60000)
    p.add_argument("--lbfgs-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lbfgs-lr", type=float, default=0.8)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--lbfgs-print-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=10.0)

    # Base losses.
    p.add_argument("--w-pde", type=float, default=1.0)
    p.add_argument("--w-div", type=float, default=5.0)
    p.add_argument("--w-near-pde", type=float, default=10.0)
    p.add_argument("--w-near-div", type=float, default=20.0)
    p.add_argument("--w-inflow-vel", type=float, default=10.0)
    p.add_argument("--w-wall", type=float, default=50.0)
    p.add_argument("--w-pressure-gauge", type=float, default=1.0)
    p.add_argument("--w-outflow-cp", type=float, default=0.0)
    p.add_argument("--w-far-cp", type=float, default=0.0)

    # Corrected Kutta / wake alignment.
    p.add_argument("--w-kutta", type=float, default=0.0)
    p.add_argument("--w-kutta-tangent", type=float, default=1.0)
    p.add_argument("--w-wake-recovery", type=float, default=0.0)
    p.add_argument("--wake-s-min", type=float, default=0.02)
    p.add_argument("--wake-s-max", type=float, default=1.0)
    p.add_argument("--wake-jitter", type=float, default=0.02)

    # XFOIL data.
    p.add_argument("--use-data", action="store_true")
    p.add_argument("--cp-data-csv", type=str, default="")
    p.add_argument("--data-alpha-filter", type=str, default="")
    p.add_argument("--w-data", type=float, default=0.0)
    p.add_argument("--data-le-boost", type=float, default=0.0)

    # Optional generation.
    p.add_argument("--make-xfoil-data", action="store_true")
    p.add_argument("--force-xfoil-data", action="store_true")
    p.add_argument("--xfoil-generator", type=str, default="generate_xfoil_cp_dataset.py")
    p.add_argument("--xfoil-outdir", type=str, default="xfoil_cp_dataset_naca2412_full")
    p.add_argument("--xfoil-alphas", type=str, default="-15,-10,-5,0,5,10,15")
    p.add_argument("--xfoil-re", type=float, default=200000.0)
    p.add_argument("--xfoil-iter", type=int, default=300)
    p.add_argument("--xfoil-inviscid", action="store_true")

    # Diagnostics.
    p.add_argument("--diag-alphas", type=str, default="-10,-5,0,5,10,15")
    p.add_argument("--sweep-alphas", type=str, default="-15,-10,-5,0,5,10,15")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    maybe_generate_xfoil(args)
    if args.use_data and not args.cp_data_csv:
        candidate = Path(args.xfoil_outdir) / "xfoil_cp_dataset.csv"
        if candidate.exists():
            args.cp_data_csv = str(candidate)
        else:
            raise ValueError("--use-data was passed but no --cp-data-csv was provided and generated CSV was not found.")
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
