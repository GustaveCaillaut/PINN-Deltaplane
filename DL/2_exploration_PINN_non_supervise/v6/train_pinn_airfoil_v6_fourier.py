import os
import math
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.path import Path


# ============================================================
# PINN Airfoil V6 fixed-case: Fourier features + pressure gauge
# ============================================================
# Goal:
#   Test whether the zero-lift solution is mostly an optimization / representation
#   issue of a plain tanh MLP.
#
# Changes vs V5:
#   1. Optional Fourier feature encoding of normalized (x,y).
#   2. Shorter warmup by default, so PDE terms act early.
#   3. More aggressive near-wall PDE/div weights by default.
#   4. Conservative learning rate by default.
#   5. Same pressure gauge strategy: Cp=0 at ONE point only.
#   6. Same default: no CL loss.
#
# Model:
#   fixed case f(x,y) -> (u,v,Cp)
#   p = q Cp, q = 0.5 U^2
#
# Convention:
#   - chord c = 1
#   - NACA 2412 centered around x=0
#   - leading edge x=-0.5, trailing edge x=+0.5
#   - freestream left -> right:
#       u_inf = U cos(alpha)
#       v_inf = U sin(alpha)
# ============================================================


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="runs_pinn_airfoil_v6_fourier")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--seed", type=int, default=1234)

# fixed case / Reynolds
parser.add_argument("--U", type=float, default=2.0)
parser.add_argument("--alpha-deg", type=float, default=5.0)
parser.add_argument("--nu", type=float, default=0.01)

# domain
parser.add_argument("--x-min", type=float, default=-4.0)
parser.add_argument("--x-max", type=float, default=6.0)
parser.add_argument("--y-min", type=float, default=-3.0)
parser.add_argument("--y-max", type=float, default=3.0)

# pressure gauge point
parser.add_argument("--x-pressure-ref", type=float, default=None)
parser.add_argument("--y-pressure-ref", type=float, default=0.0)

# NACA parameters
parser.add_argument("--naca-m", type=float, default=0.02)
parser.add_argument("--naca-p", type=float, default=0.4)
parser.add_argument("--naca-t", type=float, default=0.12)

# Fourier features
parser.add_argument(
    "--fourier-mode",
    type=str,
    default="multiscale",
    choices=["none", "random", "multiscale"],
)
parser.add_argument("--fourier-n-freqs", type=int, default=16)
parser.add_argument("--fourier-scale", type=float, default=1.0)
parser.add_argument("--fourier-include-raw", action="store_true", default=True)
parser.add_argument("--no-fourier-include-raw", dest="fourier_include_raw", action="store_false")

# model
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--depth", type=int, default=8)

# batches
parser.add_argument("--n-f", type=int, default=4096)
parser.add_argument("--n-near", type=int, default=4096)
parser.add_argument("--n-inflow", type=int, default=1024)
parser.add_argument("--n-outflow", type=int, default=1024)
parser.add_argument("--n-far-cp", type=int, default=1024)
parser.add_argument("--n-wall", type=int, default=1024)

# training
parser.add_argument("--adam-steps", type=int, default=20000)
parser.add_argument("--lbfgs-steps", type=int, default=300)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--warmup-steps", type=int, default=500)
parser.add_argument("--print-every", type=int, default=250)
parser.add_argument("--save-every", type=int, default=0)

# losses
parser.add_argument("--w-pde", type=float, default=1.0)
parser.add_argument("--w-div", type=float, default=5.0)
parser.add_argument("--w-near-pde", type=float, default=20.0)
parser.add_argument("--w-near-div", type=float, default=50.0)
parser.add_argument("--w-inflow-vel", type=float, default=10.0)
parser.add_argument("--w-wall", type=float, default=50.0)

# pressure BCs
parser.add_argument("--w-pressure-gauge", type=float, default=1.0)
parser.add_argument("--w-outflow-cp", type=float, default=0.0)
parser.add_argument("--w-far-cp", type=float, default=0.0)

# optional CL anchor, default disabled
parser.add_argument("--w-cl", type=float, default=0.0)
parser.add_argument("--cl-slope", type=float, default=4.5)
parser.add_argument("--cl-max", type=float, default=1.2)

# diagnostics
parser.add_argument("--grid-nx", type=int, default=180)
parser.add_argument("--grid-ny", type=int, default=100)

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

if args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = args.device

DTYPE = torch.float32
np.random.seed(args.seed)
torch.manual_seed(args.seed)

X_MIN, X_MAX = args.x_min, args.x_max
Y_MIN, Y_MAX = args.y_min, args.y_max
X_PRESSURE_REF = X_MAX if args.x_pressure_ref is None else float(args.x_pressure_ref)
Y_PRESSURE_REF = float(args.y_pressure_ref)

U0 = float(args.U)
ALPHA = math.radians(float(args.alpha_deg))
NU = float(args.nu)
Q0 = 0.5 * U0**2
RE = U0 / NU if NU > 0 else float("inf")

U_INF = U0 * math.cos(ALPHA)
V_INF = U0 * math.sin(ALPHA)
FREESTREAM = np.array([U_INF, V_INF], dtype=np.float64)

E_D = np.array([math.cos(ALPHA), math.sin(ALPHA)], dtype=np.float64)
E_L = np.array([-math.sin(ALPHA), math.cos(ALPHA)], dtype=np.float64)

CL_TEACHER = args.cl_max * math.tanh((args.cl_slope * ALPHA) / args.cl_max)


# ============================================================
# JSON helper
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


# ============================================================
# Geometry
# ============================================================

def naca_points(n=600, m=0.02, p=0.4, t=0.12):
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
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

    # upper leading->trailing, lower trailing->leading
    X = np.concatenate([xu, xl[::-1]])
    Y = np.concatenate([yu, yl[::-1]])
    X = X - 0.5

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
    }


AIRFOIL_X, AIRFOIL_Y = naca_points(
    n=600,
    m=args.naca_m,
    p=args.naca_p,
    t=args.naca_t,
)
AIRFOIL_PATH = Path(np.stack([AIRFOIL_X, AIRFOIL_Y], axis=1))
SURFACE = build_surface_geometry(AIRFOIL_X, AIRFOIL_Y)


def check_normals(eps=1e-3):
    pts = np.stack([SURFACE["xm"], SURFACE["ym"]], axis=1)
    n = np.stack([SURFACE["nx"], SURFACE["ny"]], axis=1)
    plus = pts + eps * n
    minus = pts - eps * n

    plus_inside = AIRFOIL_PATH.contains_points(plus)
    minus_inside = AIRFOIL_PATH.contains_points(minus)

    frac_plus_outside = float(np.mean(~plus_inside))
    frac_minus_inside = float(np.mean(minus_inside))
    ok = (frac_plus_outside > 0.95) and (frac_minus_inside > 0.95)
    return ok, frac_plus_outside, frac_minus_inside


NORMALS_OK, FRAC_PLUS_OUTSIDE, FRAC_MINUS_INSIDE = check_normals()

# Torch surface tensors
SURF_XY = torch.tensor(
    np.stack([SURFACE["xm"], SURFACE["ym"]], axis=1),
    device=DEVICE,
    dtype=DTYPE,
)
SURF_NX = torch.tensor(SURFACE["nx"].reshape(-1, 1), device=DEVICE, dtype=DTYPE)
SURF_NY = torch.tensor(SURFACE["ny"].reshape(-1, 1), device=DEVICE, dtype=DTYPE)
SURF_DS = torch.tensor(SURFACE["ds"].reshape(-1, 1), device=DEVICE, dtype=DTYPE)
E_D_T = torch.tensor(E_D.reshape(2, 1), device=DEVICE, dtype=DTYPE)
E_L_T = torch.tensor(E_L.reshape(2, 1), device=DEVICE, dtype=DTYPE)
CL_TEACHER_T = torch.tensor(CL_TEACHER, device=DEVICE, dtype=DTYPE)
PRESSURE_REF_XY = torch.tensor([[X_PRESSURE_REF, Y_PRESSURE_REF]], device=DEVICE, dtype=DTYPE)


# ============================================================
# Boundary sampling with inclined inflow/outflow
# ============================================================

BOUNDARIES = {
    "left": {
        "normal": np.array([-1.0, 0.0], dtype=np.float64),
        "is_inflow": float(np.dot(FREESTREAM, np.array([-1.0, 0.0]))) < 0,
    },
    "right": {
        "normal": np.array([1.0, 0.0], dtype=np.float64),
        "is_inflow": float(np.dot(FREESTREAM, np.array([1.0, 0.0]))) < 0,
    },
    "bottom": {
        "normal": np.array([0.0, -1.0], dtype=np.float64),
        "is_inflow": float(np.dot(FREESTREAM, np.array([0.0, -1.0]))) < 0,
    },
    "top": {
        "normal": np.array([0.0, 1.0], dtype=np.float64),
        "is_inflow": float(np.dot(FREESTREAM, np.array([0.0, 1.0]))) < 0,
    },
}

INFLOW_NAMES = [k for k, v in BOUNDARIES.items() if v["is_inflow"]]
OUTFLOW_NAMES = [k for k, v in BOUNDARIES.items() if not v["is_inflow"]]


def sample_boundary(name, n):
    if name == "left":
        x = np.full((n, 1), X_MIN, dtype=np.float32)
        y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    elif name == "right":
        x = np.full((n, 1), X_MAX, dtype=np.float32)
        y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    elif name == "bottom":
        x = np.random.uniform(X_MIN, X_MAX, size=(n, 1)).astype(np.float32)
        y = np.full((n, 1), Y_MIN, dtype=np.float32)
    elif name == "top":
        x = np.random.uniform(X_MIN, X_MAX, size=(n, 1)).astype(np.float32)
        y = np.full((n, 1), Y_MAX, dtype=np.float32)
    else:
        raise ValueError(name)
    return x, y


def sample_boundary_set(names, total_n):
    if len(names) == 0 or total_n <= 0:
        return None, None
    xs, ys = [], []
    base = total_n // len(names)
    rem = total_n - base * len(names)
    for i, name in enumerate(names):
        n = base + (1 if i < rem else 0)
        if n <= 0:
            continue
        x, y = sample_boundary(name, n)
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def sample_all_farfield(n):
    return sample_boundary_set(["left", "right", "bottom", "top"], n)


# ============================================================
# Interior / wall sampling
# ============================================================

def sample_fluid_points(n):
    pts = []
    count = 0
    while count < n:
        m = int((n - count) * 1.4) + 300
        x = np.random.uniform(X_MIN, X_MAX, size=(m, 1)).astype(np.float32)
        y = np.random.uniform(Y_MIN, Y_MAX, size=(m, 1)).astype(np.float32)
        cand = np.concatenate([x, y], axis=1)
        inside = AIRFOIL_PATH.contains_points(cand)
        fluid = cand[~inside]
        pts.append(fluid)
        count += fluid.shape[0]
    pts = np.concatenate(pts, axis=0)[:n]
    return pts[:, 0:1], pts[:, 1:2]


def sample_near_wall_points(n, r_min=0.002, r_max=0.20):
    idx = np.random.randint(0, len(SURFACE["xm"]), size=n)
    x0 = SURFACE["xm"][idx].reshape(-1, 1)
    y0 = SURFACE["ym"][idx].reshape(-1, 1)
    nx = SURFACE["nx"][idx].reshape(-1, 1)
    ny = SURFACE["ny"][idx].reshape(-1, 1)

    z = np.random.rand(n, 1).astype(np.float32)
    r = r_min * (r_max / r_min) ** z

    x = x0 + r * nx
    y = y0 + r * ny

    pts = np.concatenate([x, y], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)
    if inside.any():
        xr, yr = sample_fluid_points(int(inside.sum()))
        x[inside] = xr
        y[inside] = yr

    return x.astype(np.float32), y.astype(np.float32)


def sample_wall(n):
    ds = SURFACE["ds"]
    prob = ds / ds.sum()
    idx = np.random.choice(len(ds), size=n, replace=True, p=prob)
    x = SURFACE["xm"][idx].reshape(-1, 1)
    y = SURFACE["ym"][idx].reshape(-1, 1)
    return x.astype(np.float32), y.astype(np.float32)


def make_xy(x, y, requires_grad=False):
    arr = np.concatenate([x, y], axis=1).astype(np.float32)
    t = torch.tensor(arr, device=DEVICE, dtype=DTYPE)
    if requires_grad:
        t.requires_grad_(True)
    return t


# ============================================================
# Fourier feature model f(x,y) -> u,v,Cp
# ============================================================

class FourierEncoder(nn.Module):
    def __init__(self, mode, n_freqs, scale, include_raw):
        super().__init__()
        self.mode = mode
        self.n_freqs = int(n_freqs)
        self.scale = float(scale)
        self.include_raw = bool(include_raw)

        if mode == "none":
            self.out_dim = 2
            self.register_buffer("B", torch.empty(2, 0))
            return

        if mode == "random":
            # Random Gaussian Fourier features. Keep scale moderate for PINN derivatives.
            B = torch.randn(2, self.n_freqs, dtype=DTYPE) * self.scale
        elif mode == "multiscale":
            # Deterministic axis-aligned multi-scale frequencies.
            # n_freqs means number of scalar frequencies. We split them across x/y.
            # Example with n_freqs=16: frequencies 1,2,4,... repeated on x and y axes.
            n_each = max(1, self.n_freqs // 2)
            freqs = 2.0 ** torch.linspace(0, math.log2(max(1.0, self.scale * n_each)), n_each, dtype=DTYPE)
            Bx = torch.stack([freqs, torch.zeros_like(freqs)], dim=0)
            By = torch.stack([torch.zeros_like(freqs), freqs], dim=0)
            B = torch.cat([Bx, By], dim=1)
            # If user requested odd n_freqs, trim/pad cleanly.
            if B.shape[1] > self.n_freqs:
                B = B[:, : self.n_freqs]
            elif B.shape[1] < self.n_freqs:
                extra = torch.randn(2, self.n_freqs - B.shape[1], dtype=DTYPE) * self.scale
                B = torch.cat([B, extra], dim=1)
        else:
            raise ValueError(mode)

        self.register_buffer("B", B)
        self.out_dim = 2 * B.shape[1] + (2 if include_raw else 0)

    def forward(self, z):
        # z is normalized xy in roughly [-1,1]^2.
        if self.mode == "none":
            return z
        proj = z @ self.B
        ff = torch.cat([torch.sin(2.0 * math.pi * proj), torch.cos(2.0 * math.pi * proj)], dim=-1)
        if self.include_raw:
            return torch.cat([z, ff], dim=-1)
        return ff


class PINNFixedAirfoil(nn.Module):
    def __init__(self, width=128, depth=8):
        super().__init__()

        center = torch.tensor([
            0.5 * (X_MIN + X_MAX),
            0.5 * (Y_MIN + Y_MAX),
        ], dtype=DTYPE)
        scale = torch.tensor([
            0.5 * (X_MAX - X_MIN),
            0.5 * (Y_MAX - Y_MIN),
        ], dtype=DTYPE)

        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

        self.encoder = FourierEncoder(
            mode=args.fourier_mode,
            n_freqs=args.fourier_n_freqs,
            scale=args.fourier_scale,
            include_raw=args.fourier_include_raw,
        )

        in_dim = self.encoder.out_dim
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        z = (xy - self.center) / self.scale
        enc = self.encoder(z)
        return self.net(enc)


model = PINNFixedAirfoil(width=args.width, depth=args.depth).to(DEVICE)


# ============================================================
# Autograd residuals
# ============================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def ns_residual(model, XY):
    out = model(XY)
    u = out[:, 0:1]
    v = out[:, 1:2]
    Cp = out[:, 2:3]
    p = Q0 * Cp

    g_u = grad(u, XY)
    g_v = grad(v, XY)
    g_p = grad(p, XY)

    u_x = g_u[:, 0:1]
    u_y = g_u[:, 1:2]
    v_x = g_v[:, 0:1]
    v_y = g_v[:, 1:2]
    p_x = g_p[:, 0:1]
    p_y = g_p[:, 1:2]

    g_ux = grad(u_x, XY)
    g_uy = grad(u_y, XY)
    g_vx = grad(v_x, XY)
    g_vy = grad(v_y, XY)

    u_xx = g_ux[:, 0:1]
    u_yy = g_uy[:, 1:2]
    v_xx = g_vx[:, 0:1]
    v_yy = g_vy[:, 1:2]

    mom_x = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    div = u_x + v_y
    return mom_x, mom_y, div


def pressure_coefficients_torch():
    out = model(SURF_XY)
    Cp = out[:, 2:3]
    p = Q0 * Cp

    Fx = -torch.sum(p * SURF_NX * SURF_DS)
    Fy = -torch.sum(p * SURF_NY * SURF_DS)
    F = torch.stack([Fx, Fy]).reshape(1, 2)
    D = F @ E_D_T
    L = F @ E_L_T
    CD = D / Q0
    CL = L / Q0
    return Fx, Fy, D.squeeze(), L.squeeze(), CD.squeeze(), CL.squeeze()


# ============================================================
# Loss
# ============================================================

def compute_loss(curriculum_factor=1.0, small_batches=False):
    if small_batches:
        n_f = min(args.n_f, 2048)
        n_near = min(args.n_near, 2048)
        n_inflow = min(args.n_inflow, 512)
        n_outflow = min(args.n_outflow, 512)
        n_far_cp = min(args.n_far_cp, 512)
        n_wall = min(args.n_wall, 512)
    else:
        n_f = args.n_f
        n_near = args.n_near
        n_inflow = args.n_inflow
        n_outflow = args.n_outflow
        n_far_cp = args.n_far_cp
        n_wall = args.n_wall

    cf = float(curriculum_factor)

    # PDE global
    xf, yf = sample_fluid_points(n_f)
    XYf = make_xy(xf, yf, requires_grad=True)
    mx, my, div = ns_residual(model, XYf)
    loss_pde = torch.mean(mx**2 + my**2)
    loss_div = torch.mean(div**2)

    # PDE near wall
    xn, yn = sample_near_wall_points(n_near)
    XYn = make_xy(xn, yn, requires_grad=True)
    mxn, myn, divn = ns_residual(model, XYn)
    loss_near_pde = torch.mean(mxn**2 + myn**2)
    loss_near_div = torch.mean(divn**2)

    # Inflow velocity BC
    xi, yi = sample_boundary_set(INFLOW_NAMES, n_inflow)
    if xi is not None:
        XYi = make_xy(xi, yi)
        out_i = model(XYi)
        loss_inflow_vel = torch.mean((out_i[:, 0:1] - U_INF) ** 2 + (out_i[:, 1:2] - V_INF) ** 2)
    else:
        loss_inflow_vel = torch.tensor(0.0, device=DEVICE)

    # Optional outflow Cp BC, disabled by default
    xo, yo = sample_boundary_set(OUTFLOW_NAMES, n_outflow)
    if xo is not None:
        XYo = make_xy(xo, yo)
        Cp_out = model(XYo)[:, 2:3]
        loss_outflow_cp = torch.mean(Cp_out**2)
    else:
        loss_outflow_cp = torch.tensor(0.0, device=DEVICE)

    # Optional Cp=0 on all farfield, disabled by default
    xp, yp = sample_all_farfield(n_far_cp)
    if xp is not None:
        XYp = make_xy(xp, yp)
        Cp_far = model(XYp)[:, 2:3]
        loss_far_cp = torch.mean(Cp_far**2)
    else:
        loss_far_cp = torch.tensor(0.0, device=DEVICE)

    # Pressure gauge at one point
    Cp_ref = model(PRESSURE_REF_XY)[:, 2:3]
    loss_pressure_gauge = torch.mean(Cp_ref**2)

    # Wall no-slip
    xw, yw = sample_wall(n_wall)
    XYw = make_xy(xw, yw)
    out_w = model(XYw)
    loss_wall = torch.mean(out_w[:, 0:1] ** 2 + out_w[:, 1:2] ** 2)

    # Optional CL anchor, disabled by default
    _, _, _, _, _, CL = pressure_coefficients_torch()
    loss_cl = (CL - CL_TEACHER_T) ** 2

    loss = (
        cf * args.w_pde * loss_pde
        + cf * args.w_div * loss_div
        + cf * args.w_near_pde * loss_near_pde
        + cf * args.w_near_div * loss_near_div
        + args.w_inflow_vel * loss_inflow_vel
        + args.w_wall * loss_wall
        + args.w_pressure_gauge * loss_pressure_gauge
        + args.w_outflow_cp * loss_outflow_cp
        + args.w_far_cp * loss_far_cp
        + cf * args.w_cl * loss_cl
    )

    parts = {
        "total": loss,
        "pde": loss_pde,
        "div": loss_div,
        "near_pde": loss_near_pde,
        "near_div": loss_near_div,
        "inflow_vel": loss_inflow_vel,
        "wall": loss_wall,
        "pressure_gauge": loss_pressure_gauge,
        "outflow_cp": loss_outflow_cp,
        "far_cp": loss_far_cp,
        "cl_loss": loss_cl,
        "cl": CL.detach(),
        "cf": torch.tensor(cf, device=DEVICE),
    }
    return loss, parts


# ============================================================
# Diagnostics / saving
# ============================================================

def config_dict():
    return {
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "x_pressure_ref": X_PRESSURE_REF,
        "y_pressure_ref": Y_PRESSURE_REF,
        "U": U0,
        "alpha_deg": args.alpha_deg,
        "alpha_rad": ALPHA,
        "nu": NU,
        "Re": RE,
        "q": Q0,
        "U_inf": U_INF,
        "V_inf": V_INF,
        "naca_m": args.naca_m,
        "naca_p": args.naca_p,
        "naca_t": args.naca_t,
        "width": args.width,
        "depth": args.depth,
        "fourier_mode": args.fourier_mode,
        "fourier_n_freqs": args.fourier_n_freqs,
        "fourier_scale": args.fourier_scale,
        "fourier_include_raw": args.fourier_include_raw,
        "output": "u,v,Cp",
        "normal_check": {
            "ok": NORMALS_OK,
            "frac_plus_outside": FRAC_PLUS_OUTSIDE,
            "frac_minus_inside": FRAC_MINUS_INSIDE,
        },
        "inflow_boundaries": INFLOW_NAMES,
        "outflow_boundaries": OUTFLOW_NAMES,
        "loss_weights": {
            "w_pressure_gauge": args.w_pressure_gauge,
            "w_outflow_cp": args.w_outflow_cp,
            "w_far_cp": args.w_far_cp,
            "w_cl": args.w_cl,
            "w_near_pde": args.w_near_pde,
            "w_near_div": args.w_near_div,
        },
    }


def save_checkpoint(name, history):
    path = os.path.join(args.outdir, name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_dict(),
            "history": history,
        },
        path,
    )
    print(f"Saved {path}")


def export_onnx(name="pinn_airfoil_v6_fourier.onnx"):
    model.eval()
    dummy = torch.zeros(1, 2, device=DEVICE, dtype=DTYPE)
    path = os.path.join(args.outdir, name)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["xy"],
        output_names=["uvCp"],
        dynamic_axes={"xy": {0: "batch"}, "uvCp": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved {path}")


@torch.no_grad()
def eval_model_np(x, y):
    XY = make_xy(x, y)
    out = model(XY).cpu().numpy()
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


@torch.no_grad()
def pressure_coefficients_np():
    x = SURFACE["xm"].reshape(-1, 1)
    y = SURFACE["ym"].reshape(-1, 1)
    _, _, Cp = eval_model_np(x, y)
    p = Q0 * Cp.reshape(-1)

    nx = SURFACE["nx"].reshape(-1)
    ny = SURFACE["ny"].reshape(-1)
    ds = SURFACE["ds"].reshape(-1)

    Fx = -np.sum(p * nx * ds)
    Fy = -np.sum(p * ny * ds)
    F = np.array([Fx, Fy], dtype=np.float64)
    D = float(F @ E_D)
    L = float(F @ E_L)
    CD = D / Q0
    CL = L / Q0
    return Fx, Fy, D, L, CD, CL


def validation_residuals(n=4096):
    model.eval()
    xf, yf = sample_fluid_points(n)
    XYf = make_xy(xf, yf, requires_grad=True)
    mx, my, div = ns_residual(model, XYf)
    global_pde = float(torch.mean(mx**2 + my**2).detach().cpu())
    global_div = float(torch.mean(div**2).detach().cpu())

    xn, yn = sample_near_wall_points(n)
    XYn = make_xy(xn, yn, requires_grad=True)
    mxn, myn, divn = ns_residual(model, XYn)
    near_pde = float(torch.mean(mxn**2 + myn**2).detach().cpu())
    near_div = float(torch.mean(divn**2).detach().cpu())

    return global_pde, global_div, near_pde, near_div


def run_diagnostics(history):
    model.eval()
    diag_dir = args.outdir

    # Loss curves
    if history:
        keys = [
            "total", "pde", "div", "near_pde", "near_div", "inflow_vel",
            "wall", "pressure_gauge", "outflow_cp", "far_cp", "cl_loss"
        ]
        plt.figure(figsize=(10, 6))
        for k in keys:
            vals = np.array([h[k] for h in history], dtype=np.float64)
            plt.semilogy(vals, label=k)
        plt.grid()
        plt.legend()
        plt.xlabel("logged step")
        plt.ylabel("loss")
        plt.title("PINN V6 Fourier losses")
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, "losses.png"), dpi=150)
        plt.close()

        plt.figure()
        cls = np.array([h["cl"] for h in history], dtype=np.float64)
        plt.plot(cls, label="CL predicted")
        plt.axhline(CL_TEACHER, linestyle="--", color="k", label="CL teacher reference")
        plt.grid()
        plt.legend()
        plt.xlabel("logged step")
        plt.ylabel("CL")
        plt.title("CL during training")
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, "cl_training.png"), dpi=150)
        plt.close()

    # Normals plot
    plt.figure(figsize=(9, 3))
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    skip = max(1, len(SURFACE["xm"]) // 40)
    plt.quiver(
        SURFACE["xm"][::skip], SURFACE["ym"][::skip],
        SURFACE["nx"][::skip], SURFACE["ny"][::skip],
        angles="xy", scale_units="xy", scale=20, width=0.003,
    )
    plt.axis("equal")
    plt.grid()
    plt.title(
        f"Normals check ok={NORMALS_OK}, plus_out={FRAC_PLUS_OUTSIDE:.3f}, minus_in={FRAC_MINUS_INSIDE:.3f}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "geometry_normals.png"), dpi=150)
    plt.close()

    # Surface Cp
    x = SURFACE["xm"].reshape(-1, 1)
    y = SURFACE["ym"].reshape(-1, 1)
    _, _, Cp = eval_model_np(x, y)
    Cp = Cp.reshape(-1)
    n = len(Cp)
    half = n // 2
    Cp_upper = Cp[:half]
    Cp_lower = Cp[half:][::-1]
    x_upper = SURFACE["xm"][:half]
    x_lower = SURFACE["xm"][half:][::-1]

    plt.figure(figsize=(9, 4))
    plt.plot(x_upper, Cp_upper, label="upper Cp")
    plt.plot(x_lower, Cp_lower, label="lower Cp")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Cp")
    plt.title(f"Surface Cp, alpha={args.alpha_deg} deg, Re={RE:.1f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "surface_Cp.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(x_upper, Cp_lower - Cp_upper, label="Cp_lower - Cp_upper")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Cp jump")
    plt.title(f"Surface Cp jump, alpha={args.alpha_deg} deg, Re={RE:.1f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "surface_Cp_jump.png"), dpi=150)
    plt.close()

    # Field plots
    nxg = args.grid_nx
    nyg = args.grid_ny
    xs = np.linspace(X_MIN, X_MAX, nxg, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, nyg, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    u, v, Cp_grid = eval_model_np(pts[:, 0:1], pts[:, 1:2])
    u = u.reshape(nyg, nxg)
    v = v.reshape(nyg, nxg)
    Cp_grid = Cp_grid.reshape(nyg, nxg)
    mask = inside.reshape(nyg, nxg)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)
    Cp_grid = np.where(mask, np.nan, Cp_grid)

    plt.figure(figsize=(12, 5))
    plt.contourf(XX, YY, Cp_grid, levels=80)
    plt.colorbar(label="Cp")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.scatter([X_PRESSURE_REF], [Y_PRESSURE_REF], c="r", s=30, label="pressure gauge")
    plt.legend()
    plt.axis("equal")
    plt.title(f"Cp field, alpha={args.alpha_deg} deg, Re={RE:.1f}")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "field_Cp.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    speed = np.sqrt(u**2 + v**2)
    plt.contourf(XX, YY, speed, levels=80)
    plt.colorbar(label="speed")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Speed field, alpha={args.alpha_deg} deg, Re={RE:.1f}")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "field_speed.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    skip = (slice(None, None, max(1, nyg // 20)), slice(None, None, max(1, nxg // 30)))
    plt.quiver(XX[skip], YY[skip], u[skip], v[skip])
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Velocity field, alpha={args.alpha_deg} deg, Re={RE:.1f}")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "field_velocity.png"), dpi=150)
    plt.close()

    # Residual maps
    XY = make_xy(pts[:, 0:1].astype(np.float32), pts[:, 1:2].astype(np.float32), requires_grad=True)
    mx, my, div = ns_residual(model, XY)
    res = torch.sqrt(mx**2 + my**2).detach().cpu().numpy().reshape(nyg, nxg)
    div_np = torch.abs(div).detach().cpu().numpy().reshape(nyg, nxg)
    res = np.where(mask, np.nan, res)
    div_np = np.where(mask, np.nan, div_np)

    plt.figure(figsize=(12, 5))
    plt.contourf(XX, YY, np.log10(res + 1e-10), levels=80)
    plt.colorbar(label="log10 momentum residual")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title("Momentum residual map")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "residual_momentum_log.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.contourf(XX, YY, np.log10(div_np + 1e-10), levels=80)
    plt.colorbar(label="log10 abs(div)")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title("Divergence residual map")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "residual_divergence_log.png"), dpi=150)
    plt.close()

    Fx, Fy, D, L, CD, CL = pressure_coefficients_np()
    gpde, gdiv, npde, ndiv = validation_residuals(n=4096)

    summary = {
        "device": DEVICE,
        "U": U0,
        "alpha_deg": args.alpha_deg,
        "nu": NU,
        "Re": RE,
        "q": Q0,
        "freestream": {"u": U_INF, "v": V_INF},
        "domain": {"x_min": X_MIN, "x_max": X_MAX, "y_min": Y_MIN, "y_max": Y_MAX},
        "pressure_gauge": {"x": X_PRESSURE_REF, "y": Y_PRESSURE_REF, "weight": args.w_pressure_gauge},
        "naca": {"m": args.naca_m, "p": args.naca_p, "t": args.naca_t},
        "fourier": {
            "mode": args.fourier_mode,
            "n_freqs": args.fourier_n_freqs,
            "scale": args.fourier_scale,
            "include_raw": args.fourier_include_raw,
        },
        "normal_check": {
            "ok": NORMALS_OK,
            "frac_plus_outside": FRAC_PLUS_OUTSIDE,
            "frac_minus_inside": FRAC_MINUS_INSIDE,
        },
        "inflow_boundaries": INFLOW_NAMES,
        "outflow_boundaries": OUTFLOW_NAMES,
        "CL_teacher_reference": CL_TEACHER,
        "pressure_only": {"Fx": Fx, "Fy": Fy, "D": D, "L": L, "CD": CD, "CL": CL},
        "validation": {
            "global_pde_mse": gpde,
            "global_div_mse": gdiv,
            "near_pde_mse": npde,
            "near_div_mse": ndiv,
        },
        "loss_weights": {
            "w_cl": args.w_cl,
            "w_pressure_gauge": args.w_pressure_gauge,
            "w_outflow_cp": args.w_outflow_cp,
            "w_far_cp": args.w_far_cp,
            "w_pde": args.w_pde,
            "w_div": args.w_div,
            "w_near_pde": args.w_near_pde,
            "w_near_div": args.w_near_div,
            "w_inflow_vel": args.w_inflow_vel,
            "w_wall": args.w_wall,
        },
    }

    summary = make_json_safe(summary)

    with open(os.path.join(diag_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(diag_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("PINN V6 Fourier fixed-case summary\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")

    print("Final pressure-only coefficients:")
    print(f"  Fx={Fx:+.6e}, Fy={Fy:+.6e}")
    print(f"  D ={D:+.6e}, L ={L:+.6e}")
    print(f"  CD={CD:+.6e}, CL={CL:+.6e}, CL_teacher_ref={CL_TEACHER:+.6e}")
    print(f"  validation global_pde={gpde:.3e}, global_div={gdiv:.3e}")
    print(f"  validation near_pde={npde:.3e}, near_div={ndiv:.3e}")


# ============================================================
# Training
# ============================================================

print(f"Device: {DEVICE}")
print(f"Outdir: {args.outdir}")
print(f"Fixed case: U={U0}, alpha={args.alpha_deg} deg, nu={NU}, Re={RE:.1f}")
print(f"Freestream: u={U_INF:+.6e}, v={V_INF:+.6e}")
print(f"Domain: x=[{X_MIN},{X_MAX}], y=[{Y_MIN},{Y_MAX}]")
print(f"Pressure gauge: Cp({X_PRESSURE_REF}, {Y_PRESSURE_REF}) = 0, weight={args.w_pressure_gauge}")
print(f"NACA: m={args.naca_m}, p={args.naca_p}, t={args.naca_t}")
print(f"Fourier: mode={args.fourier_mode}, n_freqs={args.fourier_n_freqs}, scale={args.fourier_scale}, include_raw={args.fourier_include_raw}")
print(f"Airfoil signed area={SURFACE['area']:+.6e}, perimeter={SURFACE['perimeter']:.6e}")
print(f"Normal check: ok={NORMALS_OK}, plus_outside={FRAC_PLUS_OUTSIDE:.3f}, minus_inside={FRAC_MINUS_INSIDE:.3f}")
print(f"Inflow boundaries: {INFLOW_NAMES}")
print(f"Outflow boundaries: {OUTFLOW_NAMES}")
print(f"w_cl={args.w_cl}, CL_teacher_reference={CL_TEACHER:+.6e}")
print(f"w_far_cp={args.w_far_cp}, w_outflow_cp={args.w_outflow_cp}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
history = []
t0 = time.time()

for step in range(1, args.adam_steps + 1):
    optimizer.zero_grad(set_to_none=True)
    cf = min(1.0, step / max(1, args.warmup_steps))
    loss, parts = compute_loss(curriculum_factor=cf, small_batches=False)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    if step == 1 or step % args.print_every == 0:
        row = {k: float(v.detach().cpu()) for k, v in parts.items()}
        history.append(row)
        elapsed = time.time() - t0
        print(
            f"[{step:6d}/{args.adam_steps}] "
            f"loss={row['total']:.3e} | pde={row['pde']:.3e} | div={row['div']:.3e} | "
            f"near_pde={row['near_pde']:.3e} | near_div={row['near_div']:.3e} | "
            f"in_vel={row['inflow_vel']:.3e} | wall={row['wall']:.3e} | "
            f"gauge={row['pressure_gauge']:.3e} | outCp={row['outflow_cp']:.3e} | "
            f"farCp={row['far_cp']:.3e} | cl={row['cl']:+.3e} | "
            f"cl_loss={row['cl_loss']:.3e} | cf={row['cf']:.2f} | {elapsed:.1f}s"
        )

    if args.save_every > 0 and step % args.save_every == 0:
        save_checkpoint(f"pinn_airfoil_v6_fourier_step_{step}.pt", history)

save_checkpoint("pinn_airfoil_v6_fourier_adam.pt", history)

if args.lbfgs_steps > 0:
    print("Starting LBFGS refinement...")
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=20,
        max_eval=25,
        history_size=50,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    for k in range(1, args.lbfgs_steps + 1):
        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            l, _ = compute_loss(curriculum_factor=1.0, small_batches=True)
            l.backward()
            return l

        optimizer_lbfgs.step(closure)

        if k == 1 or k % 25 == 0:
            with torch.enable_grad():
                _, parts = compute_loss(curriculum_factor=1.0, small_batches=True)
            row = {kk: float(v.detach().cpu()) for kk, v in parts.items()}
            history.append(row)
            print(
                f"[LBFGS {k:4d}/{args.lbfgs_steps}] "
                f"loss={row['total']:.3e} | pde={row['pde']:.3e} | div={row['div']:.3e} | "
                f"near_pde={row['near_pde']:.3e} | near_div={row['near_div']:.3e} | "
                f"cl={row['cl']:+.3e} | gauge={row['pressure_gauge']:.3e}"
            )

save_checkpoint("pinn_airfoil_v6_fourier.pt", history)
export_onnx()
run_diagnostics(history)

print("Done.")
print(f"Artifacts saved in: {args.outdir}")
