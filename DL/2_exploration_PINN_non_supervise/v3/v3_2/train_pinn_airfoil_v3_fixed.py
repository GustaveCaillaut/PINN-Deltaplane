import os
import math
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.path import Path


# ============================================================
# PINN Airfoil V3a fixed case
# ============================================================
# Goal:
#   Before training a full parametric PINN f(x,y,U,alpha), test whether
#   a fixed-case PINN f(x,y) can produce a meaningful pressure difference
#   between upper and lower surfaces.
#
# Main changes vs V2:
#   1. Fixed U and alpha: easier diagnostic problem.
#   2. Network outputs Cp instead of p.
#          p = q Cp, q = 0.5 U^2.
#   3. Pressure reference Cp=0 imposed on all farfield boundaries.
#   4. Near-wall PDE loss kept.
#   5. Weak global CL loss from pressure integration.
#   6. Shorter default training.
#
# Convention:
#   - chord c = 1
#   - NACA 2412 centered around x=0
#   - leading edge x=-0.5, trailing edge x=+0.5
#   - freestream left -> right:
#         u_inf = U cos(alpha)
#         v_inf = U sin(alpha)
#   - alpha in radians
# ============================================================


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="runs_pinn_airfoil_v3_fixed")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--seed", type=int, default=1234)

# fixed case
parser.add_argument("--U", type=float, default=1.0)
parser.add_argument("--alpha-deg", type=float, default=5.0)
parser.add_argument("--nu", type=float, default=0.02)

# domain
parser.add_argument("--x-min", type=float, default=-1.5)
parser.add_argument("--x-max", type=float, default=2.5)
parser.add_argument("--y-min", type=float, default=-1.0)
parser.add_argument("--y-max", type=float, default=1.0)

# model
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--depth", type=int, default=8)

# batches
parser.add_argument("--n-f", type=int, default=4096)
parser.add_argument("--n-near", type=int, default=4096)
parser.add_argument("--n-inlet", type=int, default=512)
parser.add_argument("--n-far", type=int, default=512)
parser.add_argument("--n-outlet", type=int, default=512)
parser.add_argument("--n-wall", type=int, default=1024)

# training
parser.add_argument("--adam-steps", type=int, default=6000)
parser.add_argument("--lbfgs-steps", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup-steps", type=int, default=1000)
parser.add_argument("--print-every", type=int, default=200)

# losses
parser.add_argument("--w-pde", type=float, default=1.0)
parser.add_argument("--w-div", type=float, default=5.0)
parser.add_argument("--w-near-pde", type=float, default=5.0)
parser.add_argument("--w-near-div", type=float, default=20.0)
parser.add_argument("--w-inlet-vel", type=float, default=10.0)
parser.add_argument("--w-far-vel", type=float, default=5.0)
parser.add_argument("--w-far-pressure", type=float, default=5.0)
parser.add_argument("--w-wall", type=float, default=50.0)
parser.add_argument("--w-cl", type=float, default=5.0)

# analytical teacher used only for weak CL anchor
parser.add_argument("--cl-slope", type=float, default=4.5)
parser.add_argument("--cl-max", type=float, default=1.2)

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
U0 = float(args.U)
ALPHA = math.radians(float(args.alpha_deg))
NU = float(args.nu)
Q0 = 0.5 * U0**2

U_INF = U0 * math.cos(ALPHA)
V_INF = U0 * math.sin(ALPHA)

E_D = np.array([math.cos(ALPHA), math.sin(ALPHA)], dtype=np.float64)
E_L = np.array([-math.sin(ALPHA), math.cos(ALPHA)], dtype=np.float64)

CL_TEACHER = args.cl_max * math.tanh((args.cl_slope * ALPHA) / args.cl_max)


# ============================================================
# Geometry: NACA 2412
# ============================================================

def naca2412_points(n=500):
    m = 0.02
    p = 0.4
    t = 0.12

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

    # upper: leading edge -> trailing edge
    # lower: trailing edge -> leading edge
    X = np.concatenate([xu, xl[::-1]])
    Y = np.concatenate([yu, yl[::-1]])

    # center chord around x=0
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

    # outward normal for the solid boundary
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


AIRFOIL_X, AIRFOIL_Y = naca2412_points(n=500)
AIRFOIL_PATH = Path(np.stack([AIRFOIL_X, AIRFOIL_Y], axis=1))
SURFACE = build_surface_geometry(AIRFOIL_X, AIRFOIL_Y)

# Torch surface tensors for differentiable CL loss
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
CL_TEACHER_T = torch.tensor([[CL_TEACHER]], device=DEVICE, dtype=DTYPE)


# ============================================================
# Sampling
# ============================================================

def sample_fluid_points(n):
    pts = []
    count = 0

    while count < n:
        m = int((n - count) * 1.6) + 200
        x = np.random.uniform(X_MIN, X_MAX, size=(m, 1)).astype(np.float32)
        y = np.random.uniform(Y_MIN, Y_MAX, size=(m, 1)).astype(np.float32)
        cand = np.concatenate([x, y], axis=1)
        inside = AIRFOIL_PATH.contains_points(cand)
        fluid = cand[~inside]
        pts.append(fluid)
        count += fluid.shape[0]

    pts = np.concatenate(pts, axis=0)[:n]
    return pts[:, 0:1], pts[:, 1:2]


def sample_near_wall_points(n, r_min=0.002, r_max=0.12):
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


def sample_inlet(n):
    x = np.full((n, 1), X_MIN, dtype=np.float32)
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    return x, y


def sample_outlet(n):
    x = np.full((n, 1), X_MAX, dtype=np.float32)
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    return x, y


def sample_top_bottom(n):
    n_top = n // 2
    n_bot = n - n_top

    x_top = np.random.uniform(X_MIN, X_MAX, size=(n_top, 1)).astype(np.float32)
    y_top = np.full((n_top, 1), Y_MAX, dtype=np.float32)

    x_bot = np.random.uniform(X_MIN, X_MAX, size=(n_bot, 1)).astype(np.float32)
    y_bot = np.full((n_bot, 1), Y_MIN, dtype=np.float32)

    x = np.concatenate([x_top, x_bot], axis=0)
    y = np.concatenate([y_top, y_bot], axis=0)
    return x, y


def sample_all_farfield(n_inlet, n_far, n_outlet):
    xi, yi = sample_inlet(n_inlet)
    xb, yb = sample_top_bottom(n_far)
    xo, yo = sample_outlet(n_outlet)
    x = np.concatenate([xi, xb, xo], axis=0)
    y = np.concatenate([yi, yb, yo], axis=0)
    return x, y


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
# Model: f(x,y) -> (u,v,Cp)
# ============================================================

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

        layers = [nn.Linear(2, width), nn.Tanh()]
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
        return self.net(z)


model = PINNFixedAirfoil(width=args.width, depth=args.depth).to(DEVICE)


# ============================================================
# Autograd / residuals
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
        n_inlet = min(args.n_inlet, 512)
        n_far = min(args.n_far, 512)
        n_outlet = min(args.n_outlet, 512)
        n_wall = min(args.n_wall, 512)
    else:
        n_f = args.n_f
        n_near = args.n_near
        n_inlet = args.n_inlet
        n_far = args.n_far
        n_outlet = args.n_outlet
        n_wall = args.n_wall

    cf = float(curriculum_factor)

    # PDE global
    xf, yf = sample_fluid_points(n_f)
    XYf = make_xy(xf, yf, requires_grad=True)
    mx, my, div = ns_residual(model, XYf)
    loss_pde = torch.mean(mx**2 + my**2)
    loss_div = torch.mean(div**2)

    # PDE near-wall
    xn, yn = sample_near_wall_points(n_near)
    XYn = make_xy(xn, yn, requires_grad=True)
    mxn, myn, divn = ns_residual(model, XYn)
    loss_near_pde = torch.mean(mxn**2 + myn**2)
    loss_near_div = torch.mean(divn**2)

    # inlet velocity
    xi, yi = sample_inlet(n_inlet)
    XYi = make_xy(xi, yi)
    out_i = model(XYi)
    loss_inlet_vel = torch.mean((out_i[:, 0:1] - U_INF) ** 2 + (out_i[:, 1:2] - V_INF) ** 2)

    # top/bottom velocity
    xb, yb = sample_top_bottom(n_far)
    XYb = make_xy(xb, yb)
    out_b = model(XYb)
    loss_far_vel = torch.mean((out_b[:, 0:1] - U_INF) ** 2 + (out_b[:, 1:2] - V_INF) ** 2)

    # farfield pressure reference: Cp=0 on inlet + top/bottom + outlet
    xp, yp = sample_all_farfield(n_inlet, n_far, n_outlet)
    XYp = make_xy(xp, yp)
    Cp_far = model(XYp)[:, 2:3]
    loss_far_pressure = torch.mean(Cp_far**2)

    # wall no-slip
    xw, yw = sample_wall(n_wall)
    XYw = make_xy(xw, yw)
    out_w = model(XYw)
    loss_wall = torch.mean(out_w[:, 0:1] ** 2 + out_w[:, 1:2] ** 2)

    # weak global CL anchor
    _, _, _, _, _, CL = pressure_coefficients_torch()
    loss_cl = (CL - CL_TEACHER_T.squeeze()) ** 2

    loss = (
        cf * args.w_pde * loss_pde
        + cf * args.w_div * loss_div
        + cf * args.w_near_pde * loss_near_pde
        + cf * args.w_near_div * loss_near_div
        + args.w_inlet_vel * loss_inlet_vel
        + args.w_far_vel * loss_far_vel
        + args.w_far_pressure * loss_far_pressure
        + args.w_wall * loss_wall
        + cf * args.w_cl * loss_cl
    )

    parts = {
        "total": loss,
        "pde": loss_pde,
        "div": loss_div,
        "near_pde": loss_near_pde,
        "near_div": loss_near_div,
        "inlet_vel": loss_inlet_vel,
        "far_vel": loss_far_vel,
        "far_pressure": loss_far_pressure,
        "wall": loss_wall,
        "cl_loss": loss_cl,
        "cl": CL.detach(),
        "cf": torch.tensor(cf, device=DEVICE),
    }
    return loss, parts


# ============================================================
# Diagnostics / save
# ============================================================

def save_checkpoint(name, history):
    path = os.path.join(args.outdir, name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "x_min": X_MIN,
                "x_max": X_MAX,
                "y_min": Y_MIN,
                "y_max": Y_MAX,
                "U": U0,
                "alpha_deg": args.alpha_deg,
                "alpha_rad": ALPHA,
                "nu": NU,
                "q": Q0,
                "width": args.width,
                "depth": args.depth,
                "output": "u,v,Cp",
                "convention": "fixed case, freestream left_to_right",
            },
            "history": history,
        },
        path,
    )
    print(f"Saved {path}")


def export_onnx(name="pinn_airfoil_v3_fixed.onnx"):
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


def run_diagnostics(history):
    model.eval()

    # losses
    if history:
        keys = [
            "total", "pde", "div", "near_pde", "near_div", "inlet_vel",
            "far_vel", "far_pressure", "wall", "cl_loss"
        ]
        plt.figure(figsize=(10, 6))
        for k in keys:
            vals = np.array([h[k] for h in history], dtype=np.float64)
            plt.semilogy(vals, label=k)
        plt.grid()
        plt.legend()
        plt.xlabel("logged step")
        plt.ylabel("loss")
        plt.title("PINN V3 fixed losses")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "losses.png"), dpi=150)
        plt.close()

        plt.figure()
        cls = np.array([h["cl"] for h in history], dtype=np.float64)
        plt.plot(cls, label="CL predicted")
        plt.axhline(CL_TEACHER, linestyle="--", color="k", label="CL teacher")
        plt.grid()
        plt.legend()
        plt.xlabel("logged step")
        plt.ylabel("CL")
        plt.title("CL during training")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "cl_training.png"), dpi=150)
        plt.close()

    # surface pressure
    x = SURFACE["xm"].reshape(-1, 1)
    y = SURFACE["ym"].reshape(-1, 1)
    _, _, Cp = eval_model_np(x, y)
    Cp = Cp.reshape(-1)
    p = Q0 * Cp
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
    plt.title(f"Surface Cp, alpha={args.alpha_deg} deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "surface_Cp.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(x_upper, Cp_lower - Cp_upper, label="Cp_lower - Cp_upper")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Cp jump")
    plt.title(f"Surface Cp jump, alpha={args.alpha_deg} deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "surface_Cp_jump.png"), dpi=150)
    plt.close()

    # field plots
    xs = np.linspace(X_MIN, X_MAX, 180, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, 90, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    u, v, Cp_grid = eval_model_np(pts[:, 0:1], pts[:, 1:2])
    u = u.reshape(90, 180)
    v = v.reshape(90, 180)
    Cp_grid = Cp_grid.reshape(90, 180)
    mask = inside.reshape(90, 180)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)
    Cp_grid = np.where(mask, np.nan, Cp_grid)

    plt.figure(figsize=(11, 4))
    plt.contourf(XX, YY, Cp_grid, levels=60)
    plt.colorbar(label="Cp")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Cp field, alpha={args.alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "field_Cp.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 4))
    speed = np.sqrt(u**2 + v**2)
    plt.contourf(XX, YY, speed, levels=60)
    plt.colorbar(label="speed")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Speed field, alpha={args.alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "field_speed.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 4))
    skip = (slice(None, None, 5), slice(None, None, 5))
    plt.quiver(XX[skip], YY[skip], u[skip], v[skip])
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Velocity field, alpha={args.alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "field_velocity.png"), dpi=150)
    plt.close()

    # summary
    Fx, Fy, D, L, CD, CL = pressure_coefficients_np()
    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("PINN V3a fixed-case summary\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"U: {U0}\n")
        f.write(f"alpha_deg: {args.alpha_deg}\n")
        f.write(f"nu: {NU}\n")
        f.write(f"q: {Q0}\n")
        f.write(f"freestream: u={U_INF:+.6e}, v={V_INF:+.6e}\n")
        f.write(f"CL_teacher: {CL_TEACHER:+.6e}\n")
        f.write(f"surface signed area: {SURFACE['area']:+.6e}\n")
        f.write(f"surface perimeter: {SURFACE['perimeter']:+.6e}\n")
        f.write("\nPressure-only coefficients with projected directions:\n")
        f.write(f"Fx={Fx:+.6e}, Fy={Fy:+.6e}\n")
        f.write(f"D={D:+.6e}, L={L:+.6e}\n")
        f.write(f"CD_pressure={CD:+.6e}, CL_pressure={CL:+.6e}\n")
        f.write("\nNote: CD is pressure-only, viscous drag is not included.\n")

    print("Final pressure-only coefficients:")
    print(f"  Fx={Fx:+.6e}, Fy={Fy:+.6e}")
    print(f"  D ={D:+.6e}, L ={L:+.6e}")
    print(f"  CD={CD:+.6e}, CL={CL:+.6e}, CL_teacher={CL_TEACHER:+.6e}")


# ============================================================
# Training
# ============================================================

print(f"Device: {DEVICE}")
print(f"Outdir: {args.outdir}")
print(f"Fixed case: U={U0}, alpha={args.alpha_deg} deg")
print(f"Freestream: u={U_INF:+.6e}, v={V_INF:+.6e}")
print(f"CL teacher: {CL_TEACHER:+.6e}")
print(f"Airfoil area sign={SURFACE['area']:+.6e}, perimeter={SURFACE['perimeter']:+.6e}")

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
            f"in_vel={row['inlet_vel']:.3e} | far_vel={row['far_vel']:.3e} | "
            f"p_far={row['far_pressure']:.3e} | wall={row['wall']:.3e} | "
            f"cl={row['cl']:+.3e} | cl_loss={row['cl_loss']:.3e} | cf={row['cf']:.2f} | "
            f"{elapsed:.1f}s"
        )

save_checkpoint("pinn_airfoil_v3_fixed_adam.pt", history)

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
                l, parts = compute_loss(curriculum_factor=1.0, small_batches=True)
            row = {kk: float(v.detach().cpu()) for kk, v in parts.items()}
            history.append(row)
            print(
                f"[LBFGS {k:4d}/{args.lbfgs_steps}] "
                f"loss={row['total']:.3e} | pde={row['pde']:.3e} | div={row['div']:.3e} | "
                f"near_pde={row['near_pde']:.3e} | near_div={row['near_div']:.3e} | "
                f"cl={row['cl']:+.3e} | cl_loss={row['cl_loss']:.3e}"
            )

save_checkpoint("pinn_airfoil_v3_fixed.pt", history)
export_onnx()
run_diagnostics(history)

print("Done.")
print(f"Artifacts saved in: {args.outdir}")
