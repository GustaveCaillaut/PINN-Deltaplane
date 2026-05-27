"""
PINN airfoil failure laboratory, V2.

This module is designed for a reproducible notebook/experiment suite showing why
our direct PINN attempt for a NACA 2412 airfoil did not recover a lifting solution,
and why the later supervised XFOIL surrogate was needed.

Main ideas reproduced here:
  - Steady incompressible 2-D Navier--Stokes PINN, outputs u, v, p.
  - NACA 2412 obstacle, wall no-slip, freestream velocity boundary condition.
  - Pressure reference variants: gauge point vs far-field Cp = 0 on boundaries.
  - Near-wall PDE boost.
  - Fourier features.
  - Local Kutta / wake-tangency term.
  - Reynolds sweep.
  - Optional sparse/surface Cp supervision from an XFOIL CSV.

The code favors clarity and diagnostics over speed. For real reproductions, use
run_pinn_failure_suite_v2.py with --preset full or --preset overnight.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.path import Path as MplPath


# =============================================================================
# Basic utilities
# =============================================================================


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


# =============================================================================
# Geometry and surface quadrature
# =============================================================================


def naca4_surface_points(
    n_per_side: int = 450,
    m: float = 0.02,
    p: float = 0.4,
    t: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed NACA 4-digit polygon in chord-centered coordinates.

    Coordinate convention:
      LE = x=-0.5, TE=x=+0.5, mid chord x=0.
    Polygon ordering:
      upper LE->TE, then lower TE->LE.
    """
    beta = np.linspace(0.0, np.pi, n_per_side)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 0.0, 1.0))
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
        yc[~mask] = m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x[~mask] - x[~mask] ** 2)
        dyc[~mask] = 2.0 * m / (1.0 - p) ** 2 * (p - x[~mask])
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    X = np.concatenate([xu, xl[::-1]]) - 0.5
    Y = np.concatenate([yu, yl[::-1]])
    return X.astype(np.float32), Y.astype(np.float32)


def signed_area(x: np.ndarray, y: np.ndarray) -> float:
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def build_surface_quadrature(n_per_side: int = 450, m: float = 0.02, p: float = 0.4, t: float = 0.12) -> Dict:
    x, y = naca4_surface_points(n_per_side=n_per_side, m=m, p=p, t=t)
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    dx, dy = x2 - x, y2 - y
    ds = np.sqrt(dx**2 + dy**2) + 1e-12
    xm, ym = 0.5 * (x + x2), 0.5 * (y + y2)
    area = signed_area(x, y)
    # This convention is matched to the previous Python and Unreal component.
    if area < 0.0:
        nx = -dy / ds
        ny = dx / ds
    else:
        nx = dy / ds
        ny = -dx / ds
    half = len(x) // 2
    side = np.concatenate([np.ones(half, dtype=np.float32), -np.ones(len(x) - half, dtype=np.float32)])
    poly_path = MplPath(np.stack([x, y], axis=1))
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "xm": xm.astype(np.float32),
        "ym": ym.astype(np.float32),
        "nx": nx.astype(np.float32),
        "ny": ny.astype(np.float32),
        "ds": ds.astype(np.float32),
        "side": side.astype(np.float32),
        "half": int(half),
        "area": float(area),
        "perimeter": float(ds.sum()),
        "path": poly_path,
    }


def plot_airfoil_geometry(surface: Dict, ax=None, normal_stride: int = 50):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    ax.plot(surface["x"], surface["y"], "k-", lw=1.5)
    idx = np.arange(0, len(surface["xm"]), max(1, normal_stride))
    ax.quiver(surface["xm"][idx], surface["ym"][idx], surface["nx"][idx], surface["ny"][idx],
              angles="xy", scale_units="xy", scale=18, width=0.003)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x/c centered")
    ax.set_ylabel("y/c")
    ax.set_title(f"NACA 2412 quadrature, signed area={surface['area']:.4g}")
    return ax


# =============================================================================
# Sampling
# =============================================================================


def sample_interior_exterior(n: int, bounds: Tuple[float, float, float, float], path: MplPath, device, seed=None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    pts = []
    remaining = n
    while remaining > 0:
        m = max(2 * remaining, 2048)
        xy = np.column_stack([rng.uniform(xmin, xmax, size=m), rng.uniform(ymin, ymax, size=m)]).astype(np.float32)
        keep = xy[~path.contains_points(xy)]
        if len(keep) > 0:
            pts.append(keep[:remaining])
            remaining -= min(remaining, len(keep))
    arr = np.concatenate(pts, axis=0)[:n]
    return torch.tensor(arr, dtype=torch.float32, device=device)


def sample_near_airfoil(n: int, surface: Dict, bounds: Tuple[float, float, float, float], device, normal_offset_max=0.12, seed=None) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    idx = rng.integers(0, len(surface["xm"]), size=n)
    offsets = rng.uniform(0.004, normal_offset_max, size=n).astype(np.float32)
    xy = np.column_stack([
        surface["xm"][idx] + offsets * surface["nx"][idx],
        surface["ym"][idx] + offsets * surface["ny"][idx],
    ]).astype(np.float32)
    xy[:, 0] = np.clip(xy[:, 0], xmin, xmax)
    xy[:, 1] = np.clip(xy[:, 1], ymin, ymax)
    return torch.tensor(xy, dtype=torch.float32, device=device)


def sample_rectangle_boundary(n: int, bounds: Tuple[float, float, float, float], device, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boundary points and integer side code: left=0, right=1, bottom=2, top=3."""
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds
    counts = [n // 4] * 4
    for i in range(n - sum(counts)):
        counts[i] += 1
    n_left, n_right, n_bottom, n_top = counts
    left = np.column_stack([np.full(n_left, xmin), rng.uniform(ymin, ymax, size=n_left)])
    right = np.column_stack([np.full(n_right, xmax), rng.uniform(ymin, ymax, size=n_right)])
    bottom = np.column_stack([rng.uniform(xmin, xmax, size=n_bottom), np.full(n_bottom, ymin)])
    top = np.column_stack([rng.uniform(xmin, xmax, size=n_top), np.full(n_top, ymax)])
    pts = np.vstack([left, right, bottom, top]).astype(np.float32)
    sides = np.concatenate([
        np.full(n_left, 0), np.full(n_right, 1), np.full(n_bottom, 2), np.full(n_top, 3)
    ]).astype(np.int64)
    return torch.tensor(pts, dtype=torch.float32, device=device), torch.tensor(sides, dtype=torch.long, device=device)


def surface_midpoints_tensor(surface: Dict, device) -> torch.Tensor:
    xy = np.stack([surface["xm"], surface["ym"]], axis=1).astype(np.float32)
    return torch.tensor(xy, dtype=torch.float32, device=device)


# =============================================================================
# Model
# =============================================================================


class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, n_freqs=16, scale=1.0, include_raw=True):
        super().__init__()
        self.include_raw = include_raw
        # Deterministic multi-scale features: axis aligned + random leftovers.
        n_axis = min(n_freqs, 16)
        freqs = 2.0 ** torch.linspace(0.0, math.log2(max(1.0, float(scale) * max(1, n_axis // 2))), max(1, n_axis // 2))
        mats = []
        for f in freqs:
            mats.append([float(f), 0.0])
            mats.append([0.0, float(f)])
        B = torch.tensor(mats[:n_freqs], dtype=torch.float32).T
        if B.shape[1] < n_freqs:
            extra = torch.randn(in_dim, n_freqs - B.shape[1]) * scale
            B = torch.cat([B, extra], dim=1)
        self.register_buffer("B", B)
        self.out_dim = 2 * B.shape[1] + (in_dim if include_raw else 0)

    def forward(self, x):
        proj = x @ self.B
        ff = torch.cat([torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], dim=-1)
        if self.include_raw:
            return torch.cat([x, ff], dim=-1)
        return ff


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, width=128, depth=6, activation="tanh", fourier=False, fourier_freqs=16, fourier_scale=1.0):
        super().__init__()
        self.fourier = FourierFeatures(in_dim, fourier_freqs, fourier_scale) if fourier else None
        first_dim = self.fourier.out_dim if self.fourier is not None else in_dim
        act = nn.Tanh() if activation == "tanh" else (nn.SiLU() if activation == "silu" else nn.GELU())
        layers = [nn.Linear(first_dim, width), act]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        if self.fourier is not None:
            xy = self.fourier(xy)
        return self.net(xy)


# =============================================================================
# Optional XFOIL surface data
# =============================================================================


class XfoilCpDataset:
    def __init__(self, csv_path: str | Path, alpha_deg: float, U: float = 1.0):
        self.csv_path = str(csv_path)
        self.alpha_deg = float(alpha_deg)
        self.U = float(U)
        df = pd.read_csv(csv_path)
        req = {"alpha_deg", "x_pinn", "y", "Cp", "surface"}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"XFOIL CSV missing columns: {missing}")
        df = df[np.isfinite(df["alpha_deg"]) & np.isfinite(df["x_pinn"]) & np.isfinite(df["y"]) & np.isfinite(df["Cp"])]
        # Use exact alpha if available, otherwise nearest.
        unique = np.array(sorted(df["alpha_deg"].unique()), dtype=float)
        nearest = float(unique[np.argmin(np.abs(unique - alpha_deg))])
        self.alpha_used = nearest
        self.df = df[np.isclose(df["alpha_deg"], nearest)].copy().reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No XFOIL rows near alpha={alpha_deg}")
        # PINN pressure variable p is kinematic pressure scaled like U^2, with p_inf=0.
        # Cp = (p-p_inf)/(0.5 U^2) -> p_target = 0.5 U^2 Cp.
        self.df["p_target"] = 0.5 * U * U * self.df["Cp"]

    def sample(self, n: int, device, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.df), size=n)
        g = self.df.iloc[idx]
        xy = g[["x_pinn", "y"]].to_numpy(np.float32)
        p = g[["p_target"]].to_numpy(np.float32)
        return torch.tensor(xy, dtype=torch.float32, device=device), torch.tensor(p, dtype=torch.float32, device=device)


def load_xfoil_cp_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[np.isfinite(df["alpha_deg"]) & np.isfinite(df["x_pinn"]) & np.isfinite(df["Cp"])].copy()


def cp_from_xfoil_interpolated(df: pd.DataFrame, surface: Dict, alpha_deg: float) -> np.ndarray:
    g = df[np.isclose(df["alpha_deg"], alpha_deg)]
    if len(g) == 0:
        raise ValueError(f"No XFOIL data for alpha={alpha_deg}")
    cp = np.empty(len(surface["xm"]), dtype=np.float32)
    for label, side_value in [("upper", 1.0), ("lower", -1.0)]:
        gg = g[g["surface"].astype(str).str.lower().str.contains(label)].copy()
        gg = gg.groupby("x_pinn", as_index=False)["Cp"].mean().sort_values("x_pinn")
        xs = gg["x_pinn"].to_numpy(float)
        cps = gg["Cp"].to_numpy(float)
        mask = surface["side"] == side_value
        cp[mask] = np.interp(surface["xm"][mask], xs, cps, left=cps[0], right=cps[-1])
    return cp


# =============================================================================
# PDE residual and losses
# =============================================================================


@dataclass
class PINNCase:
    name: str
    alpha_deg: float = 5.0
    U: float = 1.0
    nu: float = 0.005  # Re=U/nu, chord=1.
    bounds: Tuple[float, float, float, float] = (-4.0, 6.0, -3.0, 3.0)

    n_steps: int = 4000
    lr: float = 5e-4
    width: int = 128
    depth: int = 6
    activation: str = "tanh"
    fourier: bool = False
    fourier_freqs: int = 16
    fourier_scale: float = 1.0

    n_f: int = 1024
    n_near: int = 1024
    n_bc: int = 1024
    n_wall: int = 512
    n_kutta: int = 256
    n_data: int = 512

    w_pde: float = 1.0
    w_div: float = 5.0
    w_near_pde: float = 10.0
    w_near_div: float = 20.0
    w_bc_vel: float = 10.0
    w_wall: float = 50.0
    w_gauge: float = 1.0
    w_far_cp: float = 0.0
    w_outflow_cp: float = 0.0
    w_kutta: float = 0.0
    w_data: float = 0.0

    cp_data_csv: str = ""
    pressure_target_scale: float = 1.0

    print_every: int = 500
    eval_every: int = 500
    seed: int = 1234

    @property
    def Re(self) -> float:
        return self.U / self.nu


@dataclass
class LossBreakdown:
    total: float
    pde: float
    div: float
    near_pde: float
    near_div: float
    bc_vel: float
    wall: float
    gauge: float
    far_cp: float
    outflow_cp: float
    kutta: float
    data: float


def ns_residual(model, xy: torch.Tensor, nu: float) -> Dict[str, torch.Tensor]:
    xy = xy.clone().detach().requires_grad_(True)
    out = model(xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    grad_u = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    ux, uy = grad_u[:, 0:1], grad_u[:, 1:2]
    vx, vy = grad_v[:, 0:1], grad_v[:, 1:2]
    px, py = grad_p[:, 0:1], grad_p[:, 1:2]
    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True, retain_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True, retain_graph=True)[0][:, 1:2]
    vxx = torch.autograd.grad(vx, xy, torch.ones_like(vx), create_graph=True, retain_graph=True)[0][:, 0:1]
    vyy = torch.autograd.grad(vy, xy, torch.ones_like(vy), create_graph=True, retain_graph=True)[0][:, 1:2]
    mom_x = u * ux + v * uy + px - nu * (uxx + uyy)
    mom_y = u * vx + v * vy + py - nu * (vxx + vyy)
    div = ux + vy
    return {"mom_x": mom_x, "mom_y": mom_y, "div": div, "u": u, "v": v, "p": p}


def integrate_pressure_coefficients(surface: Dict, cp: np.ndarray, alpha_deg: float, ref: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, float]:
    cp = np.asarray(cp, dtype=np.float64).reshape(-1)
    dfx = -cp * surface["nx"] * surface["ds"]
    dfy = -cp * surface["ny"] * surface["ds"]
    fx = float(np.sum(dfx))
    fy = float(np.sum(dfy))
    a = math.radians(alpha_deg)
    cd = fx * math.cos(a) + fy * math.sin(a)
    cl = -fx * math.sin(a) + fy * math.cos(a)
    rx = surface["xm"] - ref[0]
    ry = surface["ym"] - ref[1]
    cm = float(np.sum(rx * dfy - ry * dfx))
    return {"Fx_coeff": fx, "Fy_coeff": fy, "CD_pressure": cd, "CL_pressure": cl, "CM_ref": cm}


def evaluate_surface(model, surface: Dict, case: PINNCase, device, xfoil_df: Optional[pd.DataFrame] = None) -> Dict:
    model.eval()
    xy = surface_midpoints_tensor(surface, device)
    with torch.no_grad():
        out = model(xy).detach().cpu().numpy()
    p = out[:, 2].astype(np.float32)
    # Convert PINN pressure p to Cp using Cp = 2p/U^2, with gauge p_inf=0.
    cp_pred = (2.0 / (case.U * case.U)) * p
    coeffs = integrate_pressure_coefficients(surface, cp_pred, case.alpha_deg)
    result = {"uvp": out, "p": p, "cp": cp_pred, "coeffs": coeffs}
    if xfoil_df is not None:
        try:
            cp_xfoil = cp_from_xfoil_interpolated(xfoil_df, surface, case.alpha_deg)
            result["cp_xfoil"] = cp_xfoil
            result["xfoil_coeffs"] = integrate_pressure_coefficients(surface, cp_xfoil, case.alpha_deg)
            result["cp_mae_vs_xfoil"] = float(np.mean(np.abs(cp_pred - cp_xfoil)))
        except Exception as e:
            result["xfoil_error"] = str(e)
    return result


def train_case(case: PINNCase, outdir: str | Path, device="auto", surface: Optional[Dict] = None, save_model: bool = True) -> Dict:
    set_seed(case.seed)
    device = get_device(device)
    outdir = ensure_dir(outdir)
    case_dir = ensure_dir(outdir / case.name)
    surface = surface or build_surface_quadrature(n_per_side=450)

    model = MLP(width=case.width, depth=case.depth, activation=case.activation,
                fourier=case.fourier, fourier_freqs=case.fourier_freqs,
                fourier_scale=case.fourier_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=case.lr)

    alpha = math.radians(case.alpha_deg)
    u_inf = case.U * math.cos(alpha)
    v_inf = case.U * math.sin(alpha)
    gauge_xy = torch.tensor([[case.bounds[1], 0.0]], dtype=torch.float32, device=device)

    wall_xy_all = surface_midpoints_tensor(surface, device)
    if len(wall_xy_all) > case.n_wall:
        idx = torch.linspace(0, len(wall_xy_all) - 1, case.n_wall, device=device).long()
        wall_xy = wall_xy_all[idx]
    else:
        wall_xy = wall_xy_all

    xfoil_dataset = None
    xfoil_df = None
    if case.w_data > 0 and case.cp_data_csv:
        xfoil_dataset = XfoilCpDataset(case.cp_data_csv, alpha_deg=case.alpha_deg, U=case.U)
        try:
            xfoil_df = load_xfoil_cp_csv(case.cp_data_csv)
        except Exception:
            xfoil_df = None

    history: List[Dict] = []
    t0 = time.time()

    for step in range(1, case.n_steps + 1):
        opt.zero_grad(set_to_none=True)
        seed = case.seed + 100_000 * step
        xy_f = sample_interior_exterior(case.n_f, case.bounds, surface["path"], device, seed=seed)
        xy_near = sample_near_airfoil(case.n_near, surface, case.bounds, device, seed=seed + 1)
        xy_bc, bc_sides = sample_rectangle_boundary(case.n_bc, case.bounds, device, seed=seed + 2)

        rf = ns_residual(model, xy_f, case.nu)
        rnear = ns_residual(model, xy_near, case.nu)
        pde = torch.mean(rf["mom_x"] ** 2 + rf["mom_y"] ** 2)
        div = torch.mean(rf["div"] ** 2)
        near_pde = torch.mean(rnear["mom_x"] ** 2 + rnear["mom_y"] ** 2)
        near_div = torch.mean(rnear["div"] ** 2)

        bc_out = model(xy_bc)
        bc_vel = torch.mean((bc_out[:, 0:1] - u_inf) ** 2 + (bc_out[:, 1:2] - v_inf) ** 2)
        far_cp = torch.mean(bc_out[:, 2:3] ** 2)
        outflow_mask = (bc_sides == 1) | (bc_sides == 3)  # right and top: one of our tried variants.
        outflow_cp = torch.mean(bc_out[outflow_mask, 2:3] ** 2) if outflow_mask.any() else torch.tensor(0.0, device=device)

        wall_out = model(wall_xy)
        wall = torch.mean(wall_out[:, 0:1] ** 2 + wall_out[:, 1:2] ** 2)
        gauge = torch.mean(model(gauge_xy)[:, 2:3] ** 2)

        if case.w_kutta > 0 and case.n_kutta > 0:
            rng = np.random.default_rng(seed + 3)
            wake = np.column_stack([
                rng.uniform(0.52, min(case.bounds[1], 1.6), size=case.n_kutta),
                rng.uniform(-0.04, 0.04, size=case.n_kutta),
            ]).astype(np.float32)
            xy_k = torch.tensor(wake, dtype=torch.float32, device=device)
            uk = model(xy_k)[:, 0:2]
            # Wake tangency: velocity should be close to freestream direction, no equality of upper/lower Cp.
            cross = uk[:, 1:2] * math.cos(alpha) - uk[:, 0:1] * math.sin(alpha)
            kutta = torch.mean(cross ** 2)
        else:
            kutta = torch.tensor(0.0, device=device)

        if xfoil_dataset is not None and case.n_data > 0:
            xy_d, p_target = xfoil_dataset.sample(case.n_data, device, seed=seed + 4)
            p_pred = model(xy_d)[:, 2:3]
            data = torch.mean((p_pred - p_target) ** 2)
        else:
            data = torch.tensor(0.0, device=device)

        total = (
            case.w_pde * pde + case.w_div * div +
            case.w_near_pde * near_pde + case.w_near_div * near_div +
            case.w_bc_vel * bc_vel + case.w_wall * wall + case.w_gauge * gauge +
            case.w_far_cp * far_cp + case.w_outflow_cp * outflow_cp +
            case.w_kutta * kutta + case.w_data * data
        )
        total.backward()
        opt.step()

        if step == 1 or step % case.eval_every == 0 or step == case.n_steps:
            diag = evaluate_surface(model, surface, case, device, xfoil_df=xfoil_df)
            row = {
                "step": step,
                "loss": float(total.detach().cpu()),
                "pde": float(pde.detach().cpu()),
                "div": float(div.detach().cpu()),
                "near_pde": float(near_pde.detach().cpu()),
                "near_div": float(near_div.detach().cpu()),
                "bc_vel": float(bc_vel.detach().cpu()),
                "wall": float(wall.detach().cpu()),
                "gauge": float(gauge.detach().cpu()),
                "far_cp": float(far_cp.detach().cpu()),
                "outflow_cp": float(outflow_cp.detach().cpu()),
                "kutta": float(kutta.detach().cpu()),
                "data": float(data.detach().cpu()),
                "CL_pressure": diag["coeffs"]["CL_pressure"],
                "CD_pressure": diag["coeffs"]["CD_pressure"],
                "CM_ref": diag["coeffs"]["CM_ref"],
                "cp_mae_vs_xfoil": diag.get("cp_mae_vs_xfoil", np.nan),
                "time_sec": float(time.time() - t0),
            }
            history.append(row)
            print(
                f"[{case.name:>24s} {step:6d}/{case.n_steps}] "
                f"loss={row['loss']:.2e} CL={row['CL_pressure']:+.3e} "
                f"data={row['data']:.2e} pde={row['pde']:.1e} wall={row['wall']:.1e}",
                flush=True,
            )

    final = evaluate_surface(model, surface, case, device, xfoil_df=xfoil_df)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(case_dir / "history.csv", index=False)
    with open(case_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(asdict(case)), f, indent=2)
    with open(case_dir / "final_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"coeffs": final["coeffs"], "xfoil_coeffs": final.get("xfoil_coeffs"), "cp_mae_vs_xfoil": final.get("cp_mae_vs_xfoil")}), f, indent=2)
    if save_model:
        torch.save(model.state_dict(), case_dir / "model_state_dict.pt")
    make_case_plots(case_dir, case, surface, hist_df, final)
    return {"case": case, "history": hist_df, "final": final, "case_dir": case_dir, "surface": surface}


# =============================================================================
# Plotting and suite aggregation
# =============================================================================


def plot_surface_cp(surface: Dict, cp: np.ndarray, ax=None, title="Surface Cp", ylabel="Cp"):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    half = surface["half"]
    xu = surface["xm"][:half]
    xl = surface["xm"][half:][::-1]
    cpu = cp[:half]
    cpl = cp[half:][::-1]
    ax.plot(xu, cpu, label="upper")
    ax.plot(xl, cpl, label="lower")
    ax.invert_yaxis()
    ax.set_xlabel("x/c centered")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    return ax


def make_case_plots(case_dir: Path, case: PINNCase, surface: Dict, hist: pd.DataFrame, final: Dict) -> None:
    if len(hist):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.semilogy(hist["step"], hist["loss"], label="total")
        for col in ["pde", "div", "near_pde", "bc_vel", "wall", "data", "kutta"]:
            if col in hist and np.nanmax(hist[col].to_numpy(float)) > 0:
                ax.semilogy(hist["step"], hist[col], label=col, alpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(case.name)
        fig.tight_layout()
        fig.savefig(case_dir / "losses.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(hist["step"], hist["CL_pressure"], marker="o", label="CL pressure")
        if "cp_mae_vs_xfoil" in hist and hist["cp_mae_vs_xfoil"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(hist["step"], hist["cp_mae_vs_xfoil"], color="tab:red", label="Cp MAE vs XFOIL")
            ax2.set_ylabel("Cp MAE")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")
        ax.set_ylabel("CL")
        ax.set_title(case.name)
        fig.tight_layout()
        fig.savefig(case_dir / "CL_history.png", dpi=150)
        plt.close(fig)

    cp = final["cp"]
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_surface_cp(surface, cp, ax=ax, title=f"{case.name}: PINN Cp-like pressure, alpha={case.alpha_deg:g}°")
    if "cp_xfoil" in final:
        half = surface["half"]
        ax.plot(surface["xm"][:half], final["cp_xfoil"][:half], "--", color="0.3", label="XFOIL upper")
        ax.plot(surface["xm"][half:][::-1], final["cp_xfoil"][half:][::-1], "--", color="0.6", label="XFOIL lower")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(case_dir / "surface_Cp.png", dpi=150)
    plt.close(fig)

    # Simple field plots.
    try:
        xmin, xmax, ymin, ymax = case.bounds
        nx, ny = 180, 100
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        inside = surface["path"].contains_points(pts).reshape(ny, nx)
        device = next(iter([]), torch.device("cpu"))
        # Cannot recover model here because final does not store it. Field plots are produced in notebooks if model is loaded.
    except Exception:
        pass


def summarize_run_directory(outdir: str | Path) -> pd.DataFrame:
    outdir = Path(outdir)
    rows = []
    for cfg_path in sorted(outdir.glob("*/config.json")):
        case_dir = cfg_path.parent
        try:
            cfg = json.loads(cfg_path.read_text())
            hist = pd.read_csv(case_dir / "history.csv")
            diag = json.loads((case_dir / "final_diagnostics.json").read_text())
            last = hist.iloc[-1].to_dict()
            coeffs = diag.get("coeffs") or {}
            xcoeffs = diag.get("xfoil_coeffs") or {}
            rows.append({
                "case": cfg.get("name", case_dir.name),
                "Re": cfg.get("U", 1.0) / cfg.get("nu", np.nan),
                "alpha_deg": cfg.get("alpha_deg"),
                "fourier": cfg.get("fourier"),
                "w_kutta": cfg.get("w_kutta"),
                "w_data": cfg.get("w_data"),
                "w_far_cp": cfg.get("w_far_cp"),
                "CL_pressure": coeffs.get("CL_pressure"),
                "CD_pressure": coeffs.get("CD_pressure"),
                "CM_ref": coeffs.get("CM_ref"),
                "CL_xfoil": xcoeffs.get("CL_pressure"),
                "Cp_MAE_vs_XFOIL": diag.get("cp_mae_vs_xfoil"),
                "final_loss": last.get("loss"),
                "final_pde": last.get("pde"),
                "final_wall": last.get("wall"),
                "time_sec": last.get("time_sec"),
                "case_dir": str(case_dir),
            })
        except Exception as e:
            rows.append({"case": case_dir.name, "error": str(e), "case_dir": str(case_dir)})
    return pd.DataFrame(rows)


def make_suite_summary_plots(outdir: str | Path) -> pd.DataFrame:
    outdir = Path(outdir)
    df = summarize_run_directory(outdir)
    df.to_csv(outdir / "summary.csv", index=False)
    if len(df) == 0 or "CL_pressure" not in df:
        return df

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
    d = df.copy()
    d["CL_abs"] = np.abs(pd.to_numeric(d["CL_pressure"], errors="coerce"))
    d = d.sort_values("CL_abs")
    ax.barh(d["case"], d["CL_pressure"].astype(float))
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.1, color="tab:red", ls="--", lw=0.8, label="order 0.1")
    ax.axvline(-0.1, color="tab:red", ls="--", lw=0.8)
    ax.set_xlabel("CL pressure")
    ax.set_title("PINN suite: most pure variants remain near zero lift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "summary_CL_bar.png", dpi=150)
    plt.close(fig)

    if "Re" in df:
        fig, ax = plt.subplots(figsize=(8, 4))
        for label, g in df.groupby(df["w_kutta"].fillna(0).astype(float) > 0):
            name = "with Kutta" if label else "pure/no Kutta"
            gg = g.dropna(subset=["Re", "CL_pressure"]).sort_values("Re")
            if len(gg):
                ax.semilogx(gg["Re"], np.abs(gg["CL_pressure"].astype(float)), marker="o", label=name)
        ax.axhline(0.1, color="k", lw=0.8, alpha=0.6, label="order 0.1")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("Re")
        ax.set_ylabel("|CL|")
        ax.set_title("Re sweep inside the reproducible PINN suite")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / "summary_re_sweep.png", dpi=150)
        plt.close(fig)
    return df


# =============================================================================
# Case presets
# =============================================================================


def make_cases(preset: str = "full", cp_data_csv: str = "") -> List[PINNCase]:
    """Return reproducible cases.

    preset='smoke' is for checking code.
    preset='full' is a reasonable pre-run for professor discussion.
    preset='overnight' is heavier and closer to our long experiments.
    """
    if preset == "smoke":
        base = dict(n_steps=40, n_f=64, n_near=64, n_bc=64, n_wall=64, n_kutta=32, n_data=64,
                    width=32, depth=3, print_every=20, eval_every=20, lr=1e-3)
    elif preset == "overnight":
        base = dict(n_steps=12000, n_f=2048, n_near=2048, n_bc=1024, n_wall=512, n_kutta=256, n_data=1024,
                    width=128, depth=6, print_every=500, eval_every=500, lr=5e-4)
    else:
        base = dict(n_steps=5000, n_f=1024, n_near=1024, n_bc=1024, n_wall=512, n_kutta=256, n_data=512,
                    width=128, depth=6, print_every=500, eval_every=500, lr=5e-4)

    common = dict(alpha_deg=5.0, U=1.0, bounds=(-4.0, 6.0, -3.0, 3.0), **base)
    cases: List[PINNCase] = []
    # Main formulation variants at Re=200.
    cases.append(PINNCase(name="01_baseline_Re200", nu=0.005, **common))
    cases.append(PINNCase(name="02_farCp_all_boundaries", nu=0.005, w_far_cp=1.0, **common))
    cases.append(PINNCase(name="03_outflowCp_right_top", nu=0.005, w_outflow_cp=1.0, **common))
    cases.append(PINNCase(name="04_fourier_Re200", nu=0.005, fourier=True, activation="silu", fourier_freqs=24, fourier_scale=1.0, **common))
    cases.append(PINNCase(name="05_kutta_Re200", nu=0.005, w_kutta=2.0, **common))
    # Re sweep pure and Kutta.
    re_map = [(200, 0.005), (1000, 0.001), (5000, 0.0002), (20000, 0.00005)]
    for Re, nu in re_map:
        cases.append(PINNCase(name=f"10_Re{Re}_pure", nu=nu, **common))
        cases.append(PINNCase(name=f"11_Re{Re}_kutta", nu=nu, w_kutta=2.0, **common))
    # Data-prior runs, if available.
    if cp_data_csv:
        cases.append(PINNCase(name="20_xfoil_data_only_surface", nu=0.005, w_pde=0.0, w_div=0.0, w_near_pde=0.0, w_near_div=0.0,
                              w_bc_vel=0.0, w_wall=1.0, w_gauge=0.0, w_data=100.0, cp_data_csv=cp_data_csv, **common))
        cases.append(PINNCase(name="21_xfoil_data_weakPDE", nu=0.005, w_pde=0.05, w_div=0.2, w_near_pde=0.1, w_near_div=0.5,
                              w_bc_vel=1.0, w_wall=5.0, w_gauge=0.1, w_data=100.0, cp_data_csv=cp_data_csv, **common))
    return cases


def historical_re_sweep_table() -> pd.DataFrame:
    rows = [
        {"Re": 200, "variant": "pure", "CL_5deg": 7.7e-4, "CL_15deg": 1.9e-3},
        {"Re": 200, "variant": "kutta", "CL_5deg": 8.3e-4, "CL_15deg": 2.1e-3},
        {"Re": 1000, "variant": "pure", "CL_5deg": 2.3e-4, "CL_15deg": 6.1e-4},
        {"Re": 1000, "variant": "kutta", "CL_5deg": 2.4e-4, "CL_15deg": 5.9e-4},
        {"Re": 5000, "variant": "pure", "CL_5deg": -5.9e-5, "CL_15deg": 1.0e-4},
        {"Re": 5000, "variant": "kutta", "CL_5deg": 5.7e-5, "CL_15deg": 1.9e-4},
        {"Re": 20000, "variant": "pure", "CL_5deg": 4.3e-5, "CL_15deg": 1.9e-4},
        {"Re": 20000, "variant": "kutta", "CL_5deg": 6.9e-5, "CL_15deg": 1.8e-4},
    ]
    return pd.DataFrame(rows)


def historical_data_prior_table() -> pd.DataFrame:
    rows = [
        {"run": "V8 XFOIL data 200k + normal PDE", "CL_5deg": 0.447, "data_mse": 0.276, "comment": "works but PDE conflicts with data"},
        {"run": "V8 XFOIL data 200k + Kutta, Adam only", "CL_5deg": 0.43, "data_mse": 0.283, "comment": "Adam OK, LBFGS became unstable"},
        {"run": "V8 data-prior 200k + weak PDE", "CL_5deg": 0.758, "data_mse": 0.0087, "comment": "best V8 data-prior fit"},
        {"run": "V8 data-prior 20k + weak PDE", "CL_5deg": 0.249, "data_mse": 0.0024, "comment": "stable but lower lift"},
        {"run": "V9 supervised Cp surrogate", "CL_5deg": 0.805, "data_mse": 0.0012, "comment": "final chosen approach"},
    ]
    return pd.DataFrame(rows)
