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
# PINN Airfoil V7
# ============================================================
# Purpose:
#   One script to test the main branches we discussed:
#     - fixed alpha / parametric alpha / parametric alpha+U
#     - with or without Kutta-style trailing-edge loss
#     - with or without sparse XFOIL Cp surface data
#     - optional Fourier features
#     - pressure gauge only by default, not Cp=0 everywhere
#
# Model:
#   output: u, v, Cp
#   pressure used in NS residual: p = q Cp, q = 0.5 U^2
#
# Coordinate convention:
#   NACA chord c=1, centered around x=0
#   leading edge x=-0.5, trailing edge x=+0.5
#
# Freestream convention:
#   U_inf = (U cos(alpha), U sin(alpha))
#
# Recommended use:
#   Use this as a diagnostic script. Do not expect all modes to be physically
#   perfect; the point is to compare which losses/selectors escape the zero-lift
#   branch.
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
# Geometry
# ============================================================

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

    # upper: LE -> TE, lower: TE -> LE
    X = np.concatenate([xu, xl[::-1]]) - 0.5
    Y = np.concatenate([yu, yl[::-1]])
    return X.astype(np.float32), Y.astype(np.float32)


def naca4_camber_y(x, m=0.02, p=0.4):
    x = np.asarray(x)
    yc = np.zeros_like(x, dtype=float)
    if abs(m) == 0:
        return yc
    mask = x < p
    yc[mask] = m / p**2 * (2 * p * x[mask] - x[mask] ** 2)
    yc[~mask] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2)
    return yc


def naca4_y_upper_lower(x, m=0.02, p=0.4, t=0.12):
    x = np.asarray(x)
    yt = 5 * t * (
        0.2969 * np.sqrt(np.clip(x, 0, 1))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    yc = naca4_camber_y(x, m=m, p=p)
    return yc + yt, yc - yt


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

    # clockwise -> outward left normal; counterclockwise -> outward right normal
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
# XFOIL dataset generation / loading
# ============================================================

def maybe_generate_xfoil_data(args):
    if not args.make_xfoil_data:
        return
    if args.cp_data_csv and Path(args.cp_data_csv).exists() and not args.force_xfoil_data:
        print(f"Cp data already exists: {args.cp_data_csv}")
        return

    outdir = Path(args.xfoil_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    script = Path(args.xfoil_generator)
    if not script.exists():
        raise FileNotFoundError(f"Cannot find XFOIL generator: {script}")

    cmd = [
        "python",
        str(script),
        f"--alphas={args.xfoil_alphas}",
        f"--re={args.xfoil_re}",
        f"--iter={args.xfoil_iter}",
        f"--outdir={str(outdir)}",
        f"--naca-m={args.naca_m}",
        f"--naca-p={args.naca_p}",
        f"--naca-t={args.naca_t}",
    ]
    if args.xfoil_inviscid:
        cmd.append("--inviscid")
    print("Running XFOIL data generation:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def load_cp_dataset(csv_path, args):
    if csv_path is None or str(csv_path).strip() == "":
        return None
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = {"alpha_deg", "x_pinn", "y", "Cp", "surface"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cp CSV missing columns: {missing}")
    df = df.copy()
    df = df[np.isfinite(df["x_pinn"]) & np.isfinite(df["y"]) & np.isfinite(df["Cp"])]
    if args.data_alpha_filter:
        allowed = parse_float_list(args.data_alpha_filter)
        df = df[df["alpha_deg"].round(6).isin([round(a, 6) for a in allowed])]
    if len(df) == 0:
        raise ValueError("Cp dataset is empty after filtering.")
    return df.reset_index(drop=True)


# ============================================================
# Model
# ============================================================

class FourierEncoder(nn.Module):
    def __init__(self, mode, n_freqs, scale, include_raw, dtype=torch.float32):
        super().__init__()
        self.mode = mode
        self.include_raw = include_raw
        self.n_freqs = int(n_freqs)
        self.scale = float(scale)

        if mode == "none":
            self.register_buffer("B", torch.empty(2, 0, dtype=dtype))
            self.out_dim = 2
            return

        if mode == "random":
            B = torch.randn(2, self.n_freqs, dtype=dtype) * self.scale
        elif mode == "multiscale":
            n_each = max(1, self.n_freqs // 2)
            max_freq = max(1.0, self.scale * n_each)
            freqs = 2.0 ** torch.linspace(0.0, math.log2(max_freq), n_each, dtype=dtype)
            Bx = torch.stack([freqs, torch.zeros_like(freqs)], dim=0)
            By = torch.stack([torch.zeros_like(freqs), freqs], dim=0)
            B = torch.cat([Bx, By], dim=1)
            if B.shape[1] > self.n_freqs:
                B = B[:, : self.n_freqs]
            elif B.shape[1] < self.n_freqs:
                extra = torch.randn(2, self.n_freqs - B.shape[1], dtype=dtype) * self.scale
                B = torch.cat([B, extra], dim=1)
        else:
            raise ValueError(mode)

        self.register_buffer("B", B)
        self.out_dim = 2 * B.shape[1] + (2 if include_raw else 0)

    def forward(self, xy_norm):
        if self.mode == "none":
            return xy_norm
        proj = xy_norm @ self.B
        ff = torch.cat([torch.sin(2.0 * math.pi * proj), torch.cos(2.0 * math.pi * proj)], dim=-1)
        if self.include_raw:
            return torch.cat([xy_norm, ff], dim=-1)
        return ff


class AirfoilPINN(nn.Module):
    def __init__(self, args, x_min, x_max, y_min, y_max, u_scale, dtype=torch.float32):
        super().__init__()
        self.args = args
        self.param_mode = args.param_mode
        self.u_scale = float(u_scale)
        self.register_buffer("xy_center", torch.tensor([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=dtype))
        self.register_buffer("xy_scale", torch.tensor([0.5 * (x_max - x_min), 0.5 * (y_max - y_min)], dtype=dtype))
        self.encoder = FourierEncoder(args.fourier_mode, args.fourier_n_freqs, args.fourier_scale, args.fourier_include_raw, dtype=dtype)

        param_dim = 0 if args.param_mode == "fixed" else 2  # ux_norm, uy_norm
        in_dim = self.encoder.out_dim + param_dim
        layers = [nn.Linear(in_dim, args.width), nn.Tanh()]
        for _ in range(args.depth - 1):
            layers += [nn.Linear(args.width, args.width), nn.Tanh()]
        layers += [nn.Linear(args.width, 3)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inp):
        xy = inp[:, 0:2]
        xy_norm = (xy - self.xy_center) / self.xy_scale
        xy_enc = self.encoder(xy_norm)
        if self.param_mode == "fixed":
            features = xy_enc
        else:
            # input columns: x,y,ux,uy
            uv_inf = inp[:, 2:4] / self.u_scale
            features = torch.cat([xy_enc, uv_inf], dim=-1)
        return self.net(features)


# ============================================================
# Trainer
# ============================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
        self.dtype = torch.float32

        self.x_min, self.x_max = args.x_min, args.x_max
        self.y_min, self.y_max = args.y_min, args.y_max
        self.U0 = float(args.U)
        self.U_min = float(args.U_min)
        self.U_max = float(args.U_max)
        self.alpha_fixed = math.radians(float(args.alpha_deg))
        self.alpha_min = math.radians(float(args.alpha_min_deg))
        self.alpha_max = math.radians(float(args.alpha_max_deg))
        self.nu = float(args.nu)
        self.u_scale = max(abs(self.U_min), abs(self.U_max), abs(self.U0), 1e-6)

        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.airfoil_x, self.airfoil_y = naca4_points(n=args.naca_n, m=args.naca_m, p=args.naca_p, t=args.naca_t)
        self.airfoil_path = MplPath(np.stack([self.airfoil_x, self.airfoil_y], axis=1))
        self.surface = build_surface_geometry(self.airfoil_x, self.airfoil_y)
        self.normals_ok, self.frac_plus_out, self.frac_minus_in = self.check_normals()

        self.surf_xy = torch.tensor(np.stack([self.surface["xm"], self.surface["ym"]], axis=1), device=self.device, dtype=self.dtype)
        self.surf_nx = torch.tensor(self.surface["nx"].reshape(-1, 1), device=self.device, dtype=self.dtype)
        self.surf_ny = torch.tensor(self.surface["ny"].reshape(-1, 1), device=self.device, dtype=self.dtype)
        self.surf_ds = torch.tensor(self.surface["ds"].reshape(-1, 1), device=self.device, dtype=self.dtype)

        self.model = AirfoilPINN(args, self.x_min, self.x_max, self.y_min, self.y_max, self.u_scale, dtype=self.dtype).to(self.device)
        self.cp_df = load_cp_dataset(args.cp_data_csv, args) if args.use_data else None
        if self.cp_df is not None:
            print(f"Loaded Cp data: {len(self.cp_df)} rows, alphas={sorted(self.cp_df['alpha_deg'].unique())}")

        self.pressure_ref_xy = torch.tensor([[args.x_pressure_ref if args.x_pressure_ref is not None else self.x_max, args.y_pressure_ref]], device=self.device, dtype=self.dtype)
        self.history = []

    # -------------------------- params --------------------------
    def sample_params_np(self, n):
        if self.args.param_mode == "fixed":
            U = np.full((n, 1), self.U0, dtype=np.float32)
            a = np.full((n, 1), self.alpha_fixed, dtype=np.float32)
        elif self.args.param_mode == "alpha":
            U = np.full((n, 1), self.U0, dtype=np.float32)
            a = np.random.uniform(self.alpha_min, self.alpha_max, size=(n, 1)).astype(np.float32)
        elif self.args.param_mode == "alpha_u":
            U = np.random.uniform(self.U_min, self.U_max, size=(n, 1)).astype(np.float32)
            a = np.random.uniform(self.alpha_min, self.alpha_max, size=(n, 1)).astype(np.float32)
        else:
            raise ValueError(self.args.param_mode)
        ux = U * np.cos(a)
        uy = U * np.sin(a)
        return U, a, ux.astype(np.float32), uy.astype(np.float32)

    def make_input_np(self, x, y, ux, uy, requires_grad=False):
        if self.args.param_mode == "fixed":
            arr = np.concatenate([x, y], axis=1).astype(np.float32)
        else:
            arr = np.concatenate([x, y, ux, uy], axis=1).astype(np.float32)
        t = torch.tensor(arr, device=self.device, dtype=self.dtype)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def make_input_torch(self, xy, ux, uy, requires_grad=False):
        if self.args.param_mode == "fixed":
            inp = xy
        else:
            uv = torch.cat([ux, uy], dim=1)
            inp = torch.cat([xy, uv], dim=1)
        if requires_grad:
            inp.requires_grad_(True)
        return inp

    def get_ux_uy_q_from_input(self, inp):
        if self.args.param_mode == "fixed":
            ux = torch.full((inp.shape[0], 1), self.U0 * math.cos(self.alpha_fixed), device=self.device, dtype=self.dtype)
            uy = torch.full((inp.shape[0], 1), self.U0 * math.sin(self.alpha_fixed), device=self.device, dtype=self.dtype)
        else:
            ux = inp[:, 2:3]
            uy = inp[:, 3:4]
        U2 = ux**2 + uy**2
        q = 0.5 * U2
        return ux, uy, q

    # -------------------------- sampling --------------------------
    def check_normals(self, eps=1e-3):
        pts = np.stack([self.surface["xm"], self.surface["ym"]], axis=1)
        n = np.stack([self.surface["nx"], self.surface["ny"]], axis=1)
        plus = pts + eps * n
        minus = pts - eps * n
        plus_inside = self.airfoil_path.contains_points(plus)
        minus_inside = self.airfoil_path.contains_points(minus)
        frac_plus_out = float(np.mean(~plus_inside))
        frac_minus_in = float(np.mean(minus_inside))
        return (frac_plus_out > 0.95 and frac_minus_in > 0.95), frac_plus_out, frac_minus_in

    def sample_fluid_points(self, n):
        pts = []
        count = 0
        while count < n:
            m = int((n - count) * 1.4) + 300
            x = np.random.uniform(self.x_min, self.x_max, size=(m, 1)).astype(np.float32)
            y = np.random.uniform(self.y_min, self.y_max, size=(m, 1)).astype(np.float32)
            cand = np.concatenate([x, y], axis=1)
            inside = self.airfoil_path.contains_points(cand)
            fluid = cand[~inside]
            pts.append(fluid)
            count += fluid.shape[0]
        pts = np.concatenate(pts, axis=0)[:n]
        return pts[:, 0:1], pts[:, 1:2]

    def sample_near_wall_points(self, n, r_min=0.002, r_max=0.20):
        idx = np.random.randint(0, len(self.surface["xm"]), size=n)
        x0 = self.surface["xm"][idx].reshape(-1, 1)
        y0 = self.surface["ym"][idx].reshape(-1, 1)
        nx = self.surface["nx"][idx].reshape(-1, 1)
        ny = self.surface["ny"][idx].reshape(-1, 1)
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
        ds = self.surface["ds"]
        prob = ds / ds.sum()
        idx = np.random.choice(len(ds), size=n, replace=True, p=prob)
        x = self.surface["xm"][idx].reshape(-1, 1)
        y = self.surface["ym"][idx].reshape(-1, 1)
        return x.astype(np.float32), y.astype(np.float32)

    def sample_boundary(self, n):
        # Randomly sample all 4 rectangular boundaries, returning normals too.
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
            if s == 0:  # left
                x[mask] = self.x_min
                y[mask] = np.random.uniform(self.y_min, self.y_max, size=(k, 1))
                nx[mask] = -1.0
                ny[mask] = 0.0
            elif s == 1:  # right
                x[mask] = self.x_max
                y[mask] = np.random.uniform(self.y_min, self.y_max, size=(k, 1))
                nx[mask] = 1.0
                ny[mask] = 0.0
            elif s == 2:  # bottom
                x[mask] = np.random.uniform(self.x_min, self.x_max, size=(k, 1))
                y[mask] = self.y_min
                nx[mask] = 0.0
                ny[mask] = -1.0
            else:  # top
                x[mask] = np.random.uniform(self.x_min, self.x_max, size=(k, 1))
                y[mask] = self.y_max
                nx[mask] = 0.0
                ny[mask] = 1.0
        return x, y, nx, ny

    # -------------------------- residuals --------------------------
    def grad(self, outputs, inputs):
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    def ns_residual(self, inp):
        out = self.model(inp)
        u = out[:, 0:1]
        v = out[:, 1:2]
        Cp = out[:, 2:3]
        _, _, q = self.get_ux_uy_q_from_input(inp)
        p = q * Cp

        g_u = self.grad(u, inp)
        g_v = self.grad(v, inp)
        g_p = self.grad(p, inp)

        u_x = g_u[:, 0:1]
        u_y = g_u[:, 1:2]
        v_x = g_v[:, 0:1]
        v_y = g_v[:, 1:2]
        p_x = g_p[:, 0:1]
        p_y = g_p[:, 1:2]

        g_ux = self.grad(u_x, inp)
        g_uy = self.grad(u_y, inp)
        g_vx = self.grad(v_x, inp)
        g_vy = self.grad(v_y, inp)

        u_xx = g_ux[:, 0:1]
        u_yy = g_uy[:, 1:2]
        v_xx = g_vx[:, 0:1]
        v_yy = g_vy[:, 1:2]

        mom_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        mom_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        div = u_x + v_y
        return mom_x, mom_y, div

    # -------------------------- losses --------------------------
    def pressure_coefficients_torch(self, U_value=None, alpha_deg=None):
        n = self.surf_xy.shape[0]
        if U_value is None:
            U_value = self.U0
        if alpha_deg is None:
            alpha = self.alpha_fixed
        else:
            alpha = math.radians(alpha_deg)
        ux_np = np.full((n, 1), U_value * math.cos(alpha), dtype=np.float32)
        uy_np = np.full((n, 1), U_value * math.sin(alpha), dtype=np.float32)
        inp = self.make_input_np(
            self.surface["xm"].reshape(-1, 1),
            self.surface["ym"].reshape(-1, 1),
            ux_np,
            uy_np,
            requires_grad=False,
        )
        out = self.model(inp)
        Cp = out[:, 2:3]
        q = 0.5 * U_value**2
        p = q * Cp
        Fx = -torch.sum(p * self.surf_nx * self.surf_ds)
        Fy = -torch.sum(p * self.surf_ny * self.surf_ds)
        eD = torch.tensor([[math.cos(alpha)], [math.sin(alpha)]], device=self.device, dtype=self.dtype)
        eL = torch.tensor([[-math.sin(alpha)], [math.cos(alpha)]], device=self.device, dtype=self.dtype)
        F = torch.stack([Fx, Fy]).reshape(1, 2)
        D = F @ eD
        L = F @ eL
        CD = D / q
        CL = L / q
        return Fx, Fy, D.squeeze(), L.squeeze(), CD.squeeze(), CL.squeeze()

    def kutta_loss(self):
        if self.args.w_kutta <= 0:
            z = torch.tensor(0.0, device=self.device)
            return z, z, z, z

        n_pairs = self.args.n_kutta
        half = self.surface["half"]
        max_offset = min(self.args.kutta_max_offset, half - 2)
        offsets = np.random.randint(1, max_offset + 1, size=n_pairs)
        iu = half - offsets
        il = half + offsets - 1

        eps = self.args.kutta_eps
        xu = self.surface["xm"][iu].reshape(-1, 1) + eps * self.surface["nx"][iu].reshape(-1, 1)
        yu = self.surface["ym"][iu].reshape(-1, 1) + eps * self.surface["ny"][iu].reshape(-1, 1)
        xl = self.surface["xm"][il].reshape(-1, 1) + eps * self.surface["nx"][il].reshape(-1, 1)
        yl = self.surface["ym"][il].reshape(-1, 1) + eps * self.surface["ny"][il].reshape(-1, 1)

        U, a, ux, uy = self.sample_params_np(n_pairs)
        inpu = self.make_input_np(xu, yu, ux, uy)
        inpl = self.make_input_np(xl, yl, ux, uy)
        outu = self.model(inpu)
        outl = self.model(inpl)
        du = outu[:, 0:1] - outl[:, 0:1]
        dv = outu[:, 1:2] - outl[:, 1:2]
        dcp = outu[:, 2:3] - outl[:, 2:3]
        loss_cp = torch.mean(dcp**2)
        loss_vel = torch.mean(du**2 + dv**2)

        # Wake alignment: a soft local condition behind TE, not a force target.
        if self.args.w_wake_align > 0:
            s = np.random.uniform(0.02, self.args.wake_length, size=(self.args.n_wake, 1)).astype(np.float32)
            _, aw, uxw, uyw = self.sample_params_np(self.args.n_wake)
            # points behind TE along freestream direction, with small normal jitter
            U_norm = np.sqrt(uxw**2 + uyw**2) + 1e-8
            ex = uxw / U_norm
            ey = uyw / U_norm
            nx = -ey
            ny = ex
            jitter = np.random.normal(0.0, self.args.wake_jitter, size=(self.args.n_wake, 1)).astype(np.float32)
            xw = 0.5 + s * ex + jitter * nx
            yw = 0.0 + s * ey + jitter * ny
            inpw = self.make_input_np(xw, yw, uxw, uyw)
            ow = self.model(inpw)
            # zero velocity component normal to wake direction
            cross = ow[:, 0:1] * torch.tensor(ny, device=self.device, dtype=self.dtype) - ow[:, 1:2] * torch.tensor(nx, device=self.device, dtype=self.dtype)
            loss_wake = torch.mean(cross**2)
        else:
            loss_wake = torch.tensor(0.0, device=self.device)

        total = self.args.w_kutta_cp * loss_cp + self.args.w_kutta_vel * loss_vel + self.args.w_wake_align * loss_wake
        return total, loss_cp, loss_vel, loss_wake

    def data_loss(self):
        if (not self.args.use_data) or self.cp_df is None or self.args.w_data <= 0:
            z = torch.tensor(0.0, device=self.device)
            return z, z
        n = min(self.args.n_data, len(self.cp_df))
        idx = np.random.randint(0, len(self.cp_df), size=n)
        batch = self.cp_df.iloc[idx]
        x = batch["x_pinn"].to_numpy(dtype=np.float32).reshape(-1, 1)
        y = batch["y"].to_numpy(dtype=np.float32).reshape(-1, 1)
        cp_ref = batch["Cp"].to_numpy(dtype=np.float32).reshape(-1, 1)
        alpha = np.deg2rad(batch["alpha_deg"].to_numpy(dtype=np.float32).reshape(-1, 1))

        if self.args.param_mode == "fixed":
            # fixed model can only use one alpha; filtering is recommended.
            U = np.full_like(alpha, self.U0, dtype=np.float32)
            alpha[:] = self.alpha_fixed
        elif self.args.param_mode == "alpha":
            U = np.full_like(alpha, self.U0, dtype=np.float32)
        else:
            # Cp is dimensionless; reuse same data over random U as a first approximation.
            U = np.random.uniform(self.U_min, self.U_max, size=alpha.shape).astype(np.float32)

        ux = U * np.cos(alpha)
        uy = U * np.sin(alpha)
        inp = self.make_input_np(x, y, ux, uy)
        pred = self.model(inp)[:, 2:3]
        ref = torch.tensor(cp_ref, device=self.device, dtype=self.dtype)
        return torch.mean((pred - ref) ** 2), torch.mean(torch.abs(pred - ref))

    def compute_loss(self, curriculum_factor=1.0, small_batches=False):
        scale = 0.5 if small_batches else 1.0
        n_f = max(1, int(self.args.n_f * scale))
        n_near = max(1, int(self.args.n_near * scale))
        n_boundary = max(1, int(self.args.n_boundary * scale))
        n_wall = max(1, int(self.args.n_wall * scale))

        cf = float(curriculum_factor)

        # PDE global
        xf, yf = self.sample_fluid_points(n_f)
        _, _, uxf, uyf = self.sample_params_np(n_f)
        inpf = self.make_input_np(xf, yf, uxf, uyf, requires_grad=True)
        mx, my, div = self.ns_residual(inpf)
        loss_pde = torch.mean(mx**2 + my**2)
        loss_div = torch.mean(div**2)

        # PDE near wall
        xn, yn = self.sample_near_wall_points(n_near)
        _, _, uxn, uyn = self.sample_params_np(n_near)
        inpn = self.make_input_np(xn, yn, uxn, uyn, requires_grad=True)
        mxn, myn, divn = self.ns_residual(inpn)
        loss_near_pde = torch.mean(mxn**2 + myn**2)
        loss_near_div = torch.mean(divn**2)

        # boundary inflow velocity / optional outflow Cp
        xb, yb, nbx, nby = self.sample_boundary(n_boundary)
        _, _, uxb, uyb = self.sample_params_np(n_boundary)
        inpb = self.make_input_np(xb, yb, uxb, uyb)
        outb = self.model(inpb)
        dot = uxb * nbx + uyb * nby
        inflow_mask = dot.reshape(-1) < 0
        outflow_mask = ~inflow_mask
        if inflow_mask.any():
            ux_t = torch.tensor(uxb[inflow_mask], device=self.device, dtype=self.dtype)
            uy_t = torch.tensor(uyb[inflow_mask], device=self.device, dtype=self.dtype)
            loss_inflow = torch.mean((outb[inflow_mask, 0:1] - ux_t) ** 2 + (outb[inflow_mask, 1:2] - uy_t) ** 2)
        else:
            loss_inflow = torch.tensor(0.0, device=self.device)
        if outflow_mask.any():
            loss_outflow_cp = torch.mean(outb[outflow_mask, 2:3] ** 2)
        else:
            loss_outflow_cp = torch.tensor(0.0, device=self.device)
        loss_far_cp = torch.mean(outb[:, 2:3] ** 2)

        # wall no-slip
        xw, yw = self.sample_wall(n_wall)
        _, _, uxw, uyw = self.sample_params_np(n_wall)
        inpw = self.make_input_np(xw, yw, uxw, uyw)
        outw = self.model(inpw)
        loss_wall = torch.mean(outw[:, 0:1] ** 2 + outw[:, 1:2] ** 2)

        # pressure gauge
        if self.args.param_mode == "fixed":
            inp_ref = self.pressure_ref_xy
        else:
            _, _, uxg, uyg = self.sample_params_np(1)
            inp_ref = self.make_input_np(
                np.array([[self.args.x_pressure_ref if self.args.x_pressure_ref is not None else self.x_max]], dtype=np.float32),
                np.array([[self.args.y_pressure_ref]], dtype=np.float32),
                uxg,
                uyg,
            )
        loss_gauge = torch.mean(self.model(inp_ref)[:, 2:3] ** 2)

        # Kutta and data
        loss_kutta, loss_kcp, loss_kvel, loss_wake = self.kutta_loss()
        loss_data_mse, loss_data_mae = self.data_loss()

        # Optional CL anchor for control experiments only
        if self.args.w_cl > 0:
            _, _, _, _, _, CL = self.pressure_coefficients_torch(U_value=self.U0, alpha_deg=self.args.alpha_deg)
            cl_ref = self.args.cl_max * math.tanh((self.args.cl_slope * math.radians(self.args.alpha_deg)) / self.args.cl_max)
            loss_cl = (CL - torch.tensor(cl_ref, device=self.device, dtype=self.dtype)) ** 2
        else:
            _, _, _, _, _, CL = self.pressure_coefficients_torch(U_value=self.U0, alpha_deg=self.args.alpha_deg)
            loss_cl = torch.tensor(0.0, device=self.device)

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
            + self.args.w_data * loss_data_mse
            + cf * self.args.w_cl * loss_cl
        )

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
            "kutta_cp": loss_kcp,
            "kutta_vel": loss_kvel,
            "wake": loss_wake,
            "data_mse": loss_data_mse,
            "data_mae": loss_data_mae,
            "cl_loss": loss_cl,
            "cl_diag": CL.detach(),
            "cf": torch.tensor(cf, device=self.device),
        }
        return loss, parts

    # -------------------------- diagnostics --------------------------
    @torch.no_grad()
    def eval_surface_cp_np(self, U_value, alpha_deg):
        n = len(self.surface["xm"])
        a = math.radians(alpha_deg)
        ux = np.full((n, 1), U_value * math.cos(a), dtype=np.float32)
        uy = np.full((n, 1), U_value * math.sin(a), dtype=np.float32)
        inp = self.make_input_np(self.surface["xm"].reshape(-1, 1), self.surface["ym"].reshape(-1, 1), ux, uy)
        out = self.model(inp).detach().cpu().numpy()
        return out[:, 2]

    @torch.no_grad()
    def pressure_coefficients_np(self, U_value, alpha_deg):
        Cp = self.eval_surface_cp_np(U_value, alpha_deg)
        q = 0.5 * U_value**2
        p = q * Cp
        nx = self.surface["nx"]
        ny = self.surface["ny"]
        ds = self.surface["ds"]
        Fx = -float(np.sum(p * nx * ds))
        Fy = -float(np.sum(p * ny * ds))
        a = math.radians(alpha_deg)
        eD = np.array([math.cos(a), math.sin(a)])
        eL = np.array([-math.sin(a), math.cos(a)])
        F = np.array([Fx, Fy])
        D = float(F @ eD)
        L = float(F @ eL)
        return Fx, Fy, D, L, D / q, L / q

    def save_checkpoint(self, name):
        path = self.outdir / name
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config_dict(), "history": self.history}, path)
        print(f"Saved {path}")

    def config_dict(self):
        return {
            "param_mode": self.args.param_mode,
            "U": self.U0,
            "U_min": self.U_min,
            "U_max": self.U_max,
            "alpha_deg": self.args.alpha_deg,
            "alpha_min_deg": self.args.alpha_min_deg,
            "alpha_max_deg": self.args.alpha_max_deg,
            "nu": self.nu,
            "domain": [self.x_min, self.x_max, self.y_min, self.y_max],
            "naca": {"m": self.args.naca_m, "p": self.args.naca_p, "t": self.args.naca_t},
            "normal_check": {"ok": self.normals_ok, "plus_out": self.frac_plus_out, "minus_in": self.frac_minus_in},
            "fourier": {"mode": self.args.fourier_mode, "n_freqs": self.args.fourier_n_freqs, "scale": self.args.fourier_scale},
            "loss_weights": {k: getattr(self.args, k) for k in vars(self.args) if k.startswith("w_")},
            "cp_data_csv": self.args.cp_data_csv,
        }

    def run_diagnostics(self):
        self.model.eval()
        # loss curves
        if self.history:
            keys = ["total", "pde", "div", "near_pde", "near_div", "inflow", "wall", "kutta", "data_mse", "cl_diag"]
            plt.figure(figsize=(11, 6))
            for k in keys:
                vals = np.array([h[k] for h in self.history], dtype=np.float64)
                if k == "cl_diag":
                    continue
                plt.semilogy(vals + 1e-30, label=k)
            plt.grid(True)
            plt.legend()
            plt.xlabel("logged step")
            plt.ylabel("loss")
            plt.tight_layout()
            plt.savefig(self.outdir / "losses.png", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 4))
            cl_vals = np.array([h["cl_diag"] for h in self.history], dtype=np.float64)
            plt.plot(cl_vals, label=f"CL diag at alpha={self.args.alpha_deg}")
            plt.axhline(0, color="k", linewidth=0.8)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / "cl_training_diag.png", dpi=150)
            plt.close()

        # surface plots for requested alphas
        diag_alphas = parse_float_list(self.args.diag_alphas)
        if not diag_alphas:
            diag_alphas = [self.args.alpha_deg]
        coef_rows = []
        for ad in diag_alphas:
            Cp = self.eval_surface_cp_np(self.U0, ad)
            half = self.surface["half"]
            Cp_u = Cp[:half]
            Cp_l = Cp[half:][::-1]
            xu = self.surface["xm"][:half]
            xl = self.surface["xm"][half:][::-1]
            Fx, Fy, D, L, CD, CL = self.pressure_coefficients_np(self.U0, ad)
            coef_rows.append({"alpha_deg": ad, "U": self.U0, "Fx": Fx, "Fy": Fy, "D": D, "L": L, "CD": CD, "CL": CL})

            tag = f"alpha_{ad:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            plt.figure(figsize=(9, 4))
            plt.plot(xu, Cp_u, label="upper")
            plt.plot(xl, Cp_l, label="lower")
            plt.gca().invert_yaxis()
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("Cp")
            plt.title(f"surface Cp, alpha={ad:+.1f}, CL={CL:+.4f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / f"surface_Cp_{tag}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(9, 4))
            x_common = np.linspace(max(xu.min(), xl.min()), min(xu.max(), xl.max()), 250)
            c_u = np.interp(x_common, np.sort(xu), Cp_u[np.argsort(xu)])
            c_l = np.interp(x_common, np.sort(xl), Cp_l[np.argsort(xl)])
            plt.plot(x_common, c_l - c_u, label="lower - upper")
            plt.axhline(0, color="k", linewidth=0.8)
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("Cp jump")
            plt.title(f"Cp jump, alpha={ad:+.1f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / f"surface_Cp_jump_{tag}.png", dpi=150)
            plt.close()

        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(self.outdir / "coefficients.csv", index=False)

        # alpha sweep if parametric or requested
        sweep_alphas = parse_float_list(self.args.sweep_alphas)
        if sweep_alphas:
            rows = []
            for ad in sweep_alphas:
                Fx, Fy, D, L, CD, CL = self.pressure_coefficients_np(self.U0, ad)
                rows.append({"alpha_deg": ad, "U": self.U0, "Fx": Fx, "Fy": Fy, "D": D, "L": L, "CD": CD, "CL": CL})
            df = pd.DataFrame(rows)
            df.to_csv(self.outdir / "alpha_sweep.csv", index=False)
            plt.figure(figsize=(8, 4))
            plt.plot(df["alpha_deg"], df["CL"], marker="o", label="CL")
            plt.axhline(0, color="k", linewidth=0.8)
            plt.grid(True)
            plt.xlabel("alpha [deg]")
            plt.ylabel("CL")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.outdir / "alpha_sweep_CL.png", dpi=150)
            plt.close()

        # summary
        summary = {
            "config": self.config_dict(),
            "coefficients_diag": coef_rows,
            "history_last": self.history[-1] if self.history else {},
        }
        summary = make_json_safe(summary)
        with open(self.outdir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(self.outdir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, indent=2))

    # -------------------------- training --------------------------
    def train(self):
        print(f"Device: {self.device}")
        print(f"Outdir: {self.outdir}")
        print(f"param_mode={self.args.param_mode}, use_data={self.args.use_data}, w_kutta={self.args.w_kutta}, w_data={self.args.w_data}")
        print(f"normal check: ok={self.normals_ok}, plus_out={self.frac_plus_out:.3f}, minus_in={self.frac_minus_in:.3f}")

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        t0 = time.time()
        for step in range(1, self.args.adam_steps + 1):
            opt.zero_grad(set_to_none=True)
            cf = min(1.0, step / max(1, self.args.warmup_steps))
            loss, parts = self.compute_loss(curriculum_factor=cf, small_batches=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            opt.step()
            if step == 1 or step % self.args.print_every == 0:
                row = {k: float(v.detach().cpu()) for k, v in parts.items()}
                self.history.append(row)
                print(
                    f"[{step:7d}/{self.args.adam_steps}] "
                    f"loss={row['total']:.3e} pde={row['pde']:.2e} near={row['near_pde']:.2e} "
                    f"div={row['div']:.2e} wall={row['wall']:.2e} "
                    f"kutta={row['kutta']:.2e} data={row['data_mse']:.2e} "
                    f"CLdiag={row['cl_diag']:+.3e} cf={row['cf']:.2f} "
                    f"time={time.time()-t0:.1f}s"
                )
            if self.args.save_every > 0 and step % self.args.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{step}.pt")

        self.save_checkpoint("model_adam.pt")

        if self.args.lbfgs_steps > 0:
            print("Starting LBFGS...")
            opt_lbfgs = torch.optim.LBFGS(
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
                    opt_lbfgs.zero_grad(set_to_none=True)
                    l, _ = self.compute_loss(curriculum_factor=1.0, small_batches=True)
                    l.backward()
                    return l
                opt_lbfgs.step(closure)
                if k == 1 or k % self.args.lbfgs_print_every == 0:
                    _, parts = self.compute_loss(curriculum_factor=1.0, small_batches=True)
                    row = {kk: float(v.detach().cpu()) for kk, v in parts.items()}
                    self.history.append(row)
                    print(f"[LBFGS {k:5d}/{self.args.lbfgs_steps}] loss={row['total']:.3e} kutta={row['kutta']:.2e} data={row['data_mse']:.2e} CLdiag={row['cl_diag']:+.3e}")

        self.save_checkpoint("model_final.pt")
        self.run_diagnostics()
        print("Done.")


# ============================================================
# CLI
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="runs_v7")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1234)

    # parameterization
    p.add_argument("--param-mode", type=str, default="fixed", choices=["fixed", "alpha", "alpha_u"])
    p.add_argument("--U", type=float, default=2.0)
    p.add_argument("--U-min", type=float, default=1.0)
    p.add_argument("--U-max", type=float, default=3.0)
    p.add_argument("--alpha-deg", type=float, default=5.0)
    p.add_argument("--alpha-min-deg", type=float, default=-15.0)
    p.add_argument("--alpha-max-deg", type=float, default=15.0)
    p.add_argument("--nu", type=float, default=0.01)

    # domain
    p.add_argument("--x-min", type=float, default=-4.0)
    p.add_argument("--x-max", type=float, default=6.0)
    p.add_argument("--y-min", type=float, default=-3.0)
    p.add_argument("--y-max", type=float, default=3.0)
    p.add_argument("--x-pressure-ref", type=float, default=None)
    p.add_argument("--y-pressure-ref", type=float, default=0.0)

    # geometry
    p.add_argument("--naca-m", type=float, default=0.02)
    p.add_argument("--naca-p", type=float, default=0.4)
    p.add_argument("--naca-t", type=float, default=0.12)
    p.add_argument("--naca-n", type=int, default=700)

    # model
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--fourier-mode", type=str, default="multiscale", choices=["none", "random", "multiscale"])
    p.add_argument("--fourier-n-freqs", type=int, default=16)
    p.add_argument("--fourier-scale", type=float, default=1.0)
    p.add_argument("--fourier-include-raw", action="store_true", default=True)
    p.add_argument("--no-fourier-include-raw", dest="fourier_include_raw", action="store_false")

    # batches
    p.add_argument("--n-f", type=int, default=4096)
    p.add_argument("--n-near", type=int, default=4096)
    p.add_argument("--n-boundary", type=int, default=2048)
    p.add_argument("--n-wall", type=int, default=1024)
    p.add_argument("--n-data", type=int, default=1024)
    p.add_argument("--n-kutta", type=int, default=256)
    p.add_argument("--n-wake", type=int, default=256)

    # training
    p.add_argument("--adam-steps", type=int, default=30000)
    p.add_argument("--lbfgs-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lbfgs-lr", type=float, default=0.8)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--print-every", type=int, default=250)
    p.add_argument("--lbfgs-print-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=10.0)

    # base losses
    p.add_argument("--w-pde", type=float, default=1.0)
    p.add_argument("--w-div", type=float, default=5.0)
    p.add_argument("--w-near-pde", type=float, default=20.0)
    p.add_argument("--w-near-div", type=float, default=50.0)
    p.add_argument("--w-inflow-vel", type=float, default=10.0)
    p.add_argument("--w-wall", type=float, default=50.0)
    p.add_argument("--w-pressure-gauge", type=float, default=1.0)
    p.add_argument("--w-outflow-cp", type=float, default=0.0)
    p.add_argument("--w-far-cp", type=float, default=0.0)

    # Kutta / wake losses
    p.add_argument("--w-kutta", type=float, default=0.0)
    p.add_argument("--w-kutta-cp", type=float, default=1.0)
    p.add_argument("--w-kutta-vel", type=float, default=1.0)
    p.add_argument("--kutta-eps", type=float, default=0.01)
    p.add_argument("--kutta-max-offset", type=int, default=40)
    p.add_argument("--w-wake-align", type=float, default=0.0)
    p.add_argument("--wake-length", type=float, default=1.0)
    p.add_argument("--wake-jitter", type=float, default=0.02)

    # Data losses
    p.add_argument("--use-data", action="store_true")
    p.add_argument("--cp-data-csv", type=str, default="")
    p.add_argument("--data-alpha-filter", type=str, default="")
    p.add_argument("--w-data", type=float, default=0.0)

    # Optional CL control baseline
    p.add_argument("--w-cl", type=float, default=0.0)
    p.add_argument("--cl-slope", type=float, default=4.5)
    p.add_argument("--cl-max", type=float, default=1.2)

    # XFOIL generation convenience
    p.add_argument("--make-xfoil-data", action="store_true")
    p.add_argument("--force-xfoil-data", action="store_true")
    p.add_argument("--xfoil-generator", type=str, default="generate_xfoil_cp_dataset.py")
    p.add_argument("--xfoil-outdir", type=str, default="xfoil_cp_dataset_naca2412_full")
    p.add_argument("--xfoil-alphas", type=str, default="-15,-10,-5,0,5,10,15")
    p.add_argument("--xfoil-re", type=float, default=200000.0)
    p.add_argument("--xfoil-iter", type=int, default=300)
    p.add_argument("--xfoil-inviscid", action="store_true")

    # diagnostics
    p.add_argument("--diag-alphas", type=str, default="-10,-5,0,5,10,15")
    p.add_argument("--sweep-alphas", type=str, default="-15,-10,-5,0,5,10,15")
    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)
    maybe_generate_xfoil_data(args)
    # If generated and user did not pass explicit CSV, use the default generated CSV.
    if args.use_data and (not args.cp_data_csv):
        generated_csv = Path(args.xfoil_outdir) / "xfoil_cp_dataset.csv"
        if generated_csv.exists():
            args.cp_data_csv = str(generated_csv)
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
