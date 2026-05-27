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
# PINN Airfoil V2
# ============================================================
# Improvements vs V1:
#   - additional near-wall collocation points
#   - stronger incompressibility penalty
#   - simple curriculum on PDE losses
#   - optional LBFGS refinement after Adam
#   - projected CL/CD computation, not raw Fx/Fy
#   - diagnostics: near-wall residuals, surface pressure, CL/CD curves
#
# Convention:
#   - chord length c = 1
#   - NACA 2412 centered around x=0, leading edge at x=-0.5,
#     trailing edge at x=+0.5
#   - freestream left -> right:
#         u_inf = U cos(alpha)
#         v_inf = U sin(alpha)
#   - alpha in radians
# ============================================================


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default="runs_pinn_airfoil_v2")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--seed", type=int, default=1234)

# training
parser.add_argument("--adam-steps", type=int, default=20000)
parser.add_argument("--lbfgs-steps", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--print-every", type=int, default=250)
parser.add_argument("--save-every", type=int, default=5000)

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

# domain / physics
parser.add_argument("--x-min", type=float, default=-1.5)
parser.add_argument("--x-max", type=float, default=2.5)
parser.add_argument("--y-min", type=float, default=-1.0)
parser.add_argument("--y-max", type=float, default=1.0)
parser.add_argument("--u-min", type=float, default=0.5)
parser.add_argument("--u-max", type=float, default=1.5)
parser.add_argument("--alpha-max-deg", type=float, default=15.0)
parser.add_argument("--nu", type=float, default=0.02)

# loss weights
parser.add_argument("--w-pde", type=float, default=1.0)
parser.add_argument("--w-div", type=float, default=5.0)
parser.add_argument("--w-near-pde", type=float, default=5.0)
parser.add_argument("--w-near-div", type=float, default=20.0)
parser.add_argument("--w-inlet", type=float, default=10.0)
parser.add_argument("--w-far", type=float, default=5.0)
parser.add_argument("--w-outlet", type=float, default=1.0)
parser.add_argument("--w-wall", type=float, default=50.0)
parser.add_argument("--warmup-steps", type=int, default=2000)

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
U_MIN, U_MAX = args.u_min, args.u_max
ALPHA_MAX = math.radians(args.alpha_max_deg)
NU = args.nu


# ============================================================
# Geometry: NACA 2412
# ============================================================

def naca2412_points(n=400):
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

    # center chord around 0
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

    # outward normals
    if area < 0:
        nx = -dy / ds
        ny = dx / ds
    else:
        nx = dy / ds
        ny = -dx / ds

    return {
        "x": x,
        "y": y,
        "xm": xm.astype(np.float32),
        "ym": ym.astype(np.float32),
        "nx": nx.astype(np.float32),
        "ny": ny.astype(np.float32),
        "ds": ds.astype(np.float32),
        "area": float(area),
        "perimeter": float(np.sum(ds)),
    }


AIRFOIL_X, AIRFOIL_Y = naca2412_points(n=400)
AIRFOIL_PATH = Path(np.stack([AIRFOIL_X, AIRFOIL_Y], axis=1))
SURFACE = build_surface_geometry(AIRFOIL_X, AIRFOIL_Y)


# ============================================================
# Sampling
# ============================================================

def sample_params(n):
    U = np.random.uniform(U_MIN, U_MAX, size=(n, 1)).astype(np.float32)
    alpha = np.random.uniform(-ALPHA_MAX, ALPHA_MAX, size=(n, 1)).astype(np.float32)
    return U, alpha


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
    """
    Samples points just outside the airfoil along outward normals.
    This is the key addition vs V1.
    """
    idx = np.random.randint(0, len(SURFACE["xm"]), size=n)

    x0 = SURFACE["xm"][idx].reshape(-1, 1)
    y0 = SURFACE["ym"][idx].reshape(-1, 1)
    nx = SURFACE["nx"][idx].reshape(-1, 1)
    ny = SURFACE["ny"][idx].reshape(-1, 1)

    # bias toward very near wall
    z = np.random.rand(n, 1).astype(np.float32)
    r = r_min * (r_max / r_min) ** z

    x = x0 + r * nx
    y = y0 + r * ny

    pts = np.concatenate([x, y], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    # In case of numerical orientation issues, replace invalid points.
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


def sample_farfield(n):
    n_top = n // 2
    n_bot = n - n_top

    x_top = np.random.uniform(X_MIN, X_MAX, size=(n_top, 1)).astype(np.float32)
    y_top = np.full((n_top, 1), Y_MAX, dtype=np.float32)

    x_bot = np.random.uniform(X_MIN, X_MAX, size=(n_bot, 1)).astype(np.float32)
    y_bot = np.full((n_bot, 1), Y_MIN, dtype=np.float32)

    return np.concatenate([x_top, x_bot], axis=0), np.concatenate([y_top, y_bot], axis=0)


def sample_wall(n):
    ds = SURFACE["ds"]
    prob = ds / np.sum(ds)
    idx = np.random.choice(len(ds), size=n, replace=True, p=prob)
    x = SURFACE["xm"][idx].reshape(-1, 1)
    y = SURFACE["ym"][idx].reshape(-1, 1)
    return x.astype(np.float32), y.astype(np.float32)


def make_input(x, y, U, alpha, requires_grad=False):
    arr = np.concatenate([x, y, U, alpha], axis=1).astype(np.float32)
    t = torch.tensor(arr, device=DEVICE, dtype=DTYPE)
    if requires_grad:
        t.requires_grad_(True)
    return t


def freestream(U, alpha):
    return U * torch.cos(alpha), U * torch.sin(alpha)


# ============================================================
# Model
# ============================================================

class PINNAirfoil(nn.Module):
    def __init__(self, width=128, depth=8):
        super().__init__()

        center = torch.tensor([
            0.5 * (X_MIN + X_MAX),
            0.5 * (Y_MIN + Y_MAX),
            0.5 * (U_MIN + U_MAX),
            0.0,
        ], dtype=DTYPE)

        scale = torch.tensor([
            0.5 * (X_MAX - X_MIN),
            0.5 * (Y_MAX - Y_MIN),
            0.5 * (U_MAX - U_MIN),
            ALPHA_MAX,
        ], dtype=DTYPE)

        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

        layers = []
        layers.append(nn.Linear(4, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inp):
        z = (inp - self.center) / self.scale
        return self.net(z)


model = PINNAirfoil(width=args.width, depth=args.depth).to(DEVICE)


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


def ns_residual(model, X):
    out = model(X)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    g_u = grad(u, X)
    g_v = grad(v, X)
    g_p = grad(p, X)

    u_x = g_u[:, 0:1]
    u_y = g_u[:, 1:2]
    v_x = g_v[:, 0:1]
    v_y = g_v[:, 1:2]
    p_x = g_p[:, 0:1]
    p_y = g_p[:, 1:2]

    g_ux = grad(u_x, X)
    g_uy = grad(u_y, X)
    g_vx = grad(v_x, X)
    g_vy = grad(v_y, X)

    u_xx = g_ux[:, 0:1]
    u_yy = g_uy[:, 1:2]
    v_xx = g_vx[:, 0:1]
    v_yy = g_vy[:, 1:2]

    mom_x = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    div = u_x + v_y

    return mom_x, mom_y, div


# ============================================================
# Loss construction
# ============================================================

def compute_loss(batch_sizes=None, curriculum_factor=1.0):
    if batch_sizes is None:
        batch_sizes = {
            "f": args.n_f,
            "near": args.n_near,
            "inlet": args.n_inlet,
            "far": args.n_far,
            "outlet": args.n_outlet,
            "wall": args.n_wall,
        }

    # global PDE
    xf, yf = sample_fluid_points(batch_sizes["f"])
    Uf, af = sample_params(batch_sizes["f"])
    Xf = make_input(xf, yf, Uf, af, requires_grad=True)
    mx, my, div = ns_residual(model, Xf)
    loss_pde = torch.mean(mx**2 + my**2)
    loss_div = torch.mean(div**2)

    # near-wall PDE: crucial for pressure forces
    xn, yn = sample_near_wall_points(batch_sizes["near"])
    Un, an = sample_params(batch_sizes["near"])
    Xn = make_input(xn, yn, Un, an, requires_grad=True)
    mxn, myn, divn = ns_residual(model, Xn)
    loss_near_pde = torch.mean(mxn**2 + myn**2)
    loss_near_div = torch.mean(divn**2)

    # inlet BC
    xi, yi = sample_inlet(batch_sizes["inlet"])
    Ui, ai = sample_params(batch_sizes["inlet"])
    Xi = make_input(xi, yi, Ui, ai)
    out_i = model(Xi)
    ui, vi = out_i[:, 0:1], out_i[:, 1:2]
    ui_t, vi_t = freestream(Xi[:, 2:3], Xi[:, 3:4])
    loss_inlet = torch.mean((ui - ui_t) ** 2 + (vi - vi_t) ** 2)

    # farfield top/bottom BC
    xb, yb = sample_farfield(batch_sizes["far"])
    Ub, ab = sample_params(batch_sizes["far"])
    Xb = make_input(xb, yb, Ub, ab)
    out_b = model(Xb)
    ub, vb = out_b[:, 0:1], out_b[:, 1:2]
    ub_t, vb_t = freestream(Xb[:, 2:3], Xb[:, 3:4])
    loss_far = torch.mean((ub - ub_t) ** 2 + (vb - vb_t) ** 2)

    # outlet pressure reference
    xo, yo = sample_outlet(batch_sizes["outlet"])
    Uo, ao = sample_params(batch_sizes["outlet"])
    Xo = make_input(xo, yo, Uo, ao)
    po = model(Xo)[:, 2:3]
    loss_outlet = torch.mean(po**2)

    # wall no-slip
    xw, yw = sample_wall(batch_sizes["wall"])
    Uw, aw = sample_params(batch_sizes["wall"])
    Xw = make_input(xw, yw, Uw, aw)
    out_w = model(Xw)
    uw, vw = out_w[:, 0:1], out_w[:, 1:2]
    loss_wall = torch.mean(uw**2 + vw**2)

    # Curriculum: start by learning BC/wall, then progressively enforce PDE.
    cf = float(curriculum_factor)

    loss = (
        cf * args.w_pde * loss_pde
        + cf * args.w_div * loss_div
        + cf * args.w_near_pde * loss_near_pde
        + cf * args.w_near_div * loss_near_div
        + args.w_inlet * loss_inlet
        + args.w_far * loss_far
        + args.w_outlet * loss_outlet
        + args.w_wall * loss_wall
    )

    parts = {
        "total": loss,
        "pde": loss_pde,
        "div": loss_div,
        "near_pde": loss_near_pde,
        "near_div": loss_near_div,
        "inlet": loss_inlet,
        "far": loss_far,
        "outlet": loss_outlet,
        "wall": loss_wall,
        "cf": torch.tensor(cf),
    }
    return loss, parts


# ============================================================
# Save / export / diagnostics
# ============================================================

def config_dict():
    return {
        "x_min": X_MIN,
        "x_max": X_MAX,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "u_min": U_MIN,
        "u_max": U_MAX,
        "alpha_max": ALPHA_MAX,
        "nu": NU,
        "width": args.width,
        "depth": args.depth,
        "convention": "freestream_left_to_right_u=Ucosalpha_v=Usinalpha",
    }


def save_checkpoint(name, history=None):
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


def export_onnx(name="pinn_airfoil_v2.onnx"):
    model.eval()
    dummy = torch.zeros(1, 4, device=DEVICE, dtype=DTYPE)
    path = os.path.join(args.outdir, name)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["xy_U_alpha"],
        output_names=["uvp"],
        dynamic_axes={"xy_U_alpha": {0: "batch"}, "uvp": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved {path}")


@torch.no_grad()
def eval_model_np(x, y, U, alpha):
    X = make_input(x, y, U, alpha)
    out = model(X).cpu().numpy()
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


@torch.no_grad()
def surface_pressure(U_value=1.0, alpha_rad=0.0):
    x = SURFACE["xm"].reshape(-1, 1)
    y = SURFACE["ym"].reshape(-1, 1)
    U = np.full_like(x, U_value, dtype=np.float32)
    a = np.full_like(x, alpha_rad, dtype=np.float32)
    _, _, p = eval_model_np(x, y, U, a)
    return p.reshape(-1)


def projected_coefficients(U_value=1.0, alpha_deg=5.0):
    a = math.radians(alpha_deg)
    p = surface_pressure(U_value, a)
    nx = SURFACE["nx"].reshape(-1)
    ny = SURFACE["ny"].reshape(-1)
    ds = SURFACE["ds"].reshape(-1)

    Fx = -np.sum(p * nx * ds)
    Fy = -np.sum(p * ny * ds)
    F = np.array([Fx, Fy], dtype=np.float64)

    e_D = np.array([math.cos(a), math.sin(a)], dtype=np.float64)
    e_L = np.array([-math.sin(a), math.cos(a)], dtype=np.float64)

    D = float(np.dot(F, e_D))
    L = float(np.dot(F, e_L))

    q = 0.5 * U_value**2
    CD = D / q
    CL = L / q
    return Fx, Fy, D, L, CD, CL


def run_final_diagnostics(history):
    model.eval()

    # loss plot
    if history:
        keys = ["total", "pde", "div", "near_pde", "near_div", "inlet", "far", "outlet", "wall"]
        arr = {k: np.array([h[k] for h in history], dtype=np.float64) for k in keys}
        plt.figure(figsize=(9, 6))
        for k in keys:
            plt.semilogy(arr[k], label=k)
        plt.grid()
        plt.legend()
        plt.xlabel("logged step")
        plt.ylabel("loss")
        plt.title("PINN V2 losses")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "losses.png"), dpi=150)
        plt.close()

    # CL/CD projected
    alpha_degs = np.linspace(-15, 15, 61)
    rows = []
    for ad in alpha_degs:
        Fx, Fy, D, L, CD, CL = projected_coefficients(1.0, ad)
        rows.append([ad, Fx, Fy, D, L, CD, CL])
    rows = np.array(rows)
    np.savetxt(
        os.path.join(args.outdir, "alpha_sweep_projected.csv"),
        rows,
        delimiter=",",
        header="alpha_deg,Fx,Fy,D,L,CD,CL",
        comments="",
    )

    plt.figure()
    plt.plot(rows[:, 0], rows[:, 6], label="CL pressure projected")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("alpha [deg]")
    plt.ylabel("CL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cl_projected_vs_alpha.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(rows[:, 0], rows[:, 5], label="CD pressure projected")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("alpha [deg]")
    plt.ylabel("CD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cd_projected_vs_alpha.png"), dpi=150)
    plt.close()

    # surface pressure at +5 deg
    p = surface_pressure(1.0, math.radians(5.0))
    n = len(p)
    half = n // 2
    p_upper = p[:half]
    p_lower = p[half:][::-1]
    x_upper = SURFACE["xm"][:half]
    x_lower = SURFACE["xm"][half:][::-1]

    plt.figure(figsize=(9, 4))
    plt.plot(x_upper, p_upper, label="upper")
    plt.plot(x_lower, p_lower, label="lower")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("p")
    plt.legend()
    plt.title("Surface pressure, alpha=5 deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "surface_pressure_alpha_5.png"), dpi=150)
    plt.close()

    # field plot
    xs = np.linspace(X_MIN, X_MAX, 160, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, 80, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)
    U = np.full((pts.shape[0], 1), 1.0, dtype=np.float32)
    a = np.full((pts.shape[0], 1), math.radians(5.0), dtype=np.float32)
    u, v, p_grid = eval_model_np(pts[:, 0:1], pts[:, 1:2], U, a)
    p_grid = p_grid.reshape(80, 160)
    p_grid = np.where(inside.reshape(80, 160), np.nan, p_grid)

    plt.figure(figsize=(11, 4))
    plt.contourf(XX, YY, p_grid, levels=60)
    plt.colorbar(label="p")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title("Pressure field, alpha=5 deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pressure_field_alpha_5.png"), dpi=150)
    plt.close()

    # summary
    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("PINN Airfoil V2 summary\n")
        f.write(f"device: {DEVICE}\n")
        f.write(f"surface area sign: {SURFACE['area']:+.6e}\n")
        f.write(f"perimeter: {SURFACE['perimeter']:.6e}\n")
        for ad in [-10, -5, 0, 5, 10]:
            Fx, Fy, D, L, CD, CL = projected_coefficients(1.0, ad)
            f.write(
                f"alpha={ad:+.1f} deg | Fx={Fx:+.6e}, Fy={Fy:+.6e}, "
                f"D={D:+.6e}, L={L:+.6e}, CD={CD:+.6e}, CL={CL:+.6e}\n"
            )


# ============================================================
# Training: Adam then optional LBFGS
# ============================================================

print(f"Device: {DEVICE}")
print(f"Output dir: {args.outdir}")
print(f"Airfoil signed area: {SURFACE['area']:+.6e}, perimeter={SURFACE['perimeter']:.6e}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
history = []
t0 = time.time()

for step in range(1, args.adam_steps + 1):
    optimizer.zero_grad(set_to_none=True)
    cf = min(1.0, step / max(1, args.warmup_steps))
    loss, parts = compute_loss(curriculum_factor=cf)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    if step % args.print_every == 0 or step == 1:
        row = {k: float(v.detach().cpu()) for k, v in parts.items()}
        history.append(row)
        elapsed = time.time() - t0
        print(
            f"[{step:6d}/{args.adam_steps}] "
            f"loss={row['total']:.3e} | pde={row['pde']:.3e} | div={row['div']:.3e} | "
            f"near_pde={row['near_pde']:.3e} | near_div={row['near_div']:.3e} | "
            f"in={row['inlet']:.3e} | far={row['far']:.3e} | out={row['outlet']:.3e} | "
            f"wall={row['wall']:.3e} | cf={row['cf']:.2f} | {elapsed:.1f}s"
        )

    if args.save_every > 0 and step % args.save_every == 0:
        save_checkpoint(f"pinn_airfoil_v2_step_{step}.pt", history=history)

save_checkpoint("pinn_airfoil_v2_adam.pt", history=history)

# LBFGS refinement. Uses smaller batches because each LBFGS step calls closure multiple times.
if args.lbfgs_steps > 0:
    print("Starting LBFGS refinement...")
    lbfgs_batches = {
        "f": min(args.n_f, 2048),
        "near": min(args.n_near, 2048),
        "inlet": min(args.n_inlet, 512),
        "far": min(args.n_far, 512),
        "outlet": min(args.n_outlet, 512),
        "wall": min(args.n_wall, 512),
    }

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
            l, _ = compute_loss(batch_sizes=lbfgs_batches, curriculum_factor=1.0)
            l.backward()
            return l

        loss_val = optimizer_lbfgs.step(closure)

        if k % 25 == 0 or k == 1:
            with torch.enable_grad():
                l, parts = compute_loss(batch_sizes=lbfgs_batches, curriculum_factor=1.0)
            row = {kk: float(v.detach().cpu()) for kk, v in parts.items()}
            history.append(row)
            print(
                f"[LBFGS {k:4d}/{args.lbfgs_steps}] "
                f"loss={row['total']:.3e} | pde={row['pde']:.3e} | div={row['div']:.3e} | "
                f"near_pde={row['near_pde']:.3e} | near_div={row['near_div']:.3e} | wall={row['wall']:.3e}"
            )

save_checkpoint("pinn_airfoil_v2.pt", history=history)
export_onnx("pinn_airfoil_v2.onnx")
run_final_diagnostics(history)

print("Done.")
print(f"Artifacts saved in: {args.outdir}")
