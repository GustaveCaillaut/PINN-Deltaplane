import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.path import Path


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="pinn_airfoil_v1.pt")
parser.add_argument("--outdir", type=str, default="diag_pinn_v1")
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--depth", type=int, default=7)
parser.add_argument("--n-val", type=int, default=4096)
parser.add_argument("--u", type=float, default=1.0)
parser.add_argument("--alpha-deg", type=float, default=5.0)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

if args.device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = args.device

DTYPE = torch.float32
torch.manual_seed(1234)
np.random.seed(1234)


# ============================================================
# Load config
# ============================================================

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
config = ckpt.get("config", {})

X_MIN = config.get("x_min", -1.5)
X_MAX = config.get("x_max", 2.5)
Y_MIN = config.get("y_min", -1.0)
Y_MAX = config.get("y_max", 1.0)
U_MIN = config.get("u_min", 0.5)
U_MAX = config.get("u_max", 1.5)
ALPHA_MAX = config.get("alpha_max", math.radians(15.0))
NU = config.get("nu", 0.02)


# ============================================================
# Geometry
# ============================================================

def naca2412_points(n=300):
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
    yc[mask] = m / p**2 * (2.0 * p * x[mask] - x[mask]**2)
    dyc[mask] = 2.0 * m / p**2 * (p - x[mask])

    yc[~mask] = m / (1.0 - p)**2 * (
        (1.0 - 2.0 * p) + 2.0 * p * x[~mask] - x[~mask]**2
    )
    dyc[~mask] = 2.0 * m / (1.0 - p)**2 * (p - x[~mask])

    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

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
        "x": x,
        "y": y,
        "xm": xm.astype(np.float32),
        "ym": ym.astype(np.float32),
        "nx": nx.astype(np.float32),
        "ny": ny.astype(np.float32),
        "ds": ds.astype(np.float32),
        "area": area,
    }


AIRFOIL_X, AIRFOIL_Y = naca2412_points(n=300)
AIRFOIL_PATH = Path(np.stack([AIRFOIL_X, AIRFOIL_Y], axis=1))
SURFACE = build_surface_geometry(AIRFOIL_X, AIRFOIL_Y)


# ============================================================
# Model
# ============================================================

class PINNAirfoil(nn.Module):
    def __init__(self, width=128, depth=7):
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

    def forward(self, inp):
        z = (inp - self.center) / self.scale
        return self.net(z)


model = PINNAirfoil(width=args.width, depth=args.depth).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


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
        m = int((n - count) * 1.5) + 100
        x = np.random.uniform(X_MIN, X_MAX, size=(m, 1)).astype(np.float32)
        y = np.random.uniform(Y_MIN, Y_MAX, size=(m, 1)).astype(np.float32)
        cand = np.concatenate([x, y], axis=1)
        inside = AIRFOIL_PATH.contains_points(cand)
        fluid = cand[~inside]
        pts.append(fluid)
        count += fluid.shape[0]
    pts = np.concatenate(pts, axis=0)[:n]
    return pts[:, 0:1], pts[:, 1:2]


def sample_near_wall_points(n, radius=0.08):
    """
    Points proches de l'aile, utiles pour savoir si le PINN est mauvais localement.
    """
    idx = np.random.randint(0, len(SURFACE["xm"]), size=n)

    x0 = SURFACE["xm"][idx].reshape(-1, 1)
    y0 = SURFACE["ym"][idx].reshape(-1, 1)
    nx = SURFACE["nx"][idx].reshape(-1, 1)
    ny = SURFACE["ny"][idx].reshape(-1, 1)

    r = np.random.uniform(0.005, radius, size=(n, 1)).astype(np.float32)

    x = x0 + r * nx
    y = y0 + r * ny

    pts = np.concatenate([x, y], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    # sécurité : si certains tombent dans l'aile à cause de normales inversées/locales,
    # on les remplace par des points fluides normaux.
    if inside.any():
        xr, yr = sample_fluid_points(int(inside.sum()))
        x[inside] = xr
        y[inside] = yr

    return x.astype(np.float32), y.astype(np.float32)


def make_input_np(x, y, U, alpha, requires_grad=False):
    arr = np.concatenate([x, y, U, alpha], axis=1).astype(np.float32)
    t = torch.tensor(arr, device=DEVICE, dtype=DTYPE)
    if requires_grad:
        t.requires_grad_(True)
    return t


def freestream_torch(U, alpha):
    u_inf = U * torch.cos(alpha)
    v_inf = U * torch.sin(alpha)
    return u_inf, v_inf


def freestream_np(U, alpha):
    return U * np.cos(alpha), U * np.sin(alpha)


# ============================================================
# Autograd PDE
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
# Diagnostics helpers
# ============================================================

@torch.no_grad()
def eval_model_np(x, y, U, alpha):
    X = make_input_np(x, y, U, alpha, requires_grad=False)
    out = model(X).cpu().numpy()
    return out[:, 0:1], out[:, 1:2], out[:, 2:3]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def mae(x):
    return float(np.mean(np.abs(np.asarray(x))))


def summarize_array(name, arr):
    arr = np.asarray(arr)
    return (
        f"{name}: mean={arr.mean():+.4e}, std={arr.std():.4e}, "
        f"min={arr.min():+.4e}, max={arr.max():+.4e}, rms={rms(arr):.4e}"
    )


# ============================================================
# 1. Geometry diagnostics
# ============================================================

def plot_geometry():
    plt.figure(figsize=(9, 3))
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k-", linewidth=2, label="airfoil")
    skip = 20
    plt.quiver(
        SURFACE["xm"][::skip],
        SURFACE["ym"][::skip],
        SURFACE["nx"][::skip],
        SURFACE["ny"][::skip],
        angles="xy",
        scale_units="xy",
        scale=20,
        width=0.003,
        label="normals",
    )
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.title(f"Airfoil geometry and normals, signed area={SURFACE['area']:.4e}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "geometry_normals.png"), dpi=150)
    plt.close()


# ============================================================
# 2. Validation losses
# ============================================================

def validation_metrics(n):
    metrics = {}

    # interior random
    x, y = sample_fluid_points(n)
    U, a = sample_params(n)
    X = make_input_np(x, y, U, a, requires_grad=True)

    mx, my, div = ns_residual(model, X)

    metrics["pde_mse"] = float(torch.mean(mx**2 + my**2).detach().cpu())
    metrics["mom_x_rms"] = float(torch.sqrt(torch.mean(mx**2)).detach().cpu())
    metrics["mom_y_rms"] = float(torch.sqrt(torch.mean(my**2)).detach().cpu())
    metrics["div_mse"] = float(torch.mean(div**2).detach().cpu())
    metrics["div_rms"] = float(torch.sqrt(torch.mean(div**2)).detach().cpu())

    # near wall PDE
    x, y = sample_near_wall_points(n, radius=0.08)
    U, a = sample_params(n)
    X = make_input_np(x, y, U, a, requires_grad=True)

    mx, my, div = ns_residual(model, X)

    metrics["near_wall_pde_mse"] = float(torch.mean(mx**2 + my**2).detach().cpu())
    metrics["near_wall_div_mse"] = float(torch.mean(div**2).detach().cpu())
    metrics["near_wall_div_rms"] = float(torch.sqrt(torch.mean(div**2)).detach().cpu())

    # inlet
    x = np.full((n, 1), X_MIN, dtype=np.float32)
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    U, a = sample_params(n)

    up, vp, pp = eval_model_np(x, y, U, a)
    ut, vt = freestream_np(U, a)

    metrics["inlet_u_rmse"] = rms(up - ut)
    metrics["inlet_v_rmse"] = rms(vp - vt)
    metrics["inlet_speed_rmse"] = rms(np.sqrt((up - ut)**2 + (vp - vt)**2))

    # far top/bottom
    n_top = n // 2
    n_bot = n - n_top
    x_top = np.random.uniform(X_MIN, X_MAX, size=(n_top, 1)).astype(np.float32)
    y_top = np.full((n_top, 1), Y_MAX, dtype=np.float32)
    x_bot = np.random.uniform(X_MIN, X_MAX, size=(n_bot, 1)).astype(np.float32)
    y_bot = np.full((n_bot, 1), Y_MIN, dtype=np.float32)
    x = np.concatenate([x_top, x_bot], axis=0)
    y = np.concatenate([y_top, y_bot], axis=0)
    U, a = sample_params(n)

    up, vp, pp = eval_model_np(x, y, U, a)
    ut, vt = freestream_np(U, a)

    metrics["far_u_rmse"] = rms(up - ut)
    metrics["far_v_rmse"] = rms(vp - vt)
    metrics["far_speed_rmse"] = rms(np.sqrt((up - ut)**2 + (vp - vt)**2))

    # outlet pressure
    x = np.full((n, 1), X_MAX, dtype=np.float32)
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    U, a = sample_params(n)

    up, vp, pp = eval_model_np(x, y, U, a)
    metrics["outlet_p_rms"] = rms(pp)
    metrics["outlet_p_mean"] = float(pp.mean())

    # wall no slip
    idx = np.random.randint(0, len(SURFACE["xm"]), size=n)
    x = SURFACE["xm"][idx].reshape(-1, 1)
    y = SURFACE["ym"][idx].reshape(-1, 1)
    U, a = sample_params(n)

    up, vp, pp = eval_model_np(x, y, U, a)
    wall_speed = np.sqrt(up**2 + vp**2)

    metrics["wall_u_rms"] = rms(up)
    metrics["wall_v_rms"] = rms(vp)
    metrics["wall_speed_rms"] = rms(wall_speed)
    metrics["wall_speed_max"] = float(wall_speed.max())

    return metrics


# ============================================================
# 3. Surface pressure and pressure force
# ============================================================

@torch.no_grad()
def surface_pressure(U_value=1.0, alpha_deg=5.0):
    alpha_value = math.radians(alpha_deg)

    x = SURFACE["xm"].reshape(-1, 1)
    y = SURFACE["ym"].reshape(-1, 1)
    U = np.full_like(x, U_value, dtype=np.float32)
    a = np.full_like(x, alpha_value, dtype=np.float32)

    _, _, p = eval_model_np(x, y, U, a)

    return p.reshape(-1)


def pressure_force(U_value=1.0, alpha_deg=5.0):
    p = surface_pressure(U_value, alpha_deg)

    nx = SURFACE["nx"].reshape(-1)
    ny = SURFACE["ny"].reshape(-1)
    ds = SURFACE["ds"].reshape(-1)

    fx = -np.sum(p * nx * ds)
    fy = -np.sum(p * ny * ds)

    q = 0.5 * U_value**2
    c_ref = 1.0

    cd_p = fx / (q * c_ref)
    cl_p = fy / (q * c_ref)

    return fx, fy, cl_p, cd_p


def alpha_sweep():
    alphas = np.linspace(-15, 15, 61)
    rows = []

    for a_deg in alphas:
        fx, fy, clp, cdp = pressure_force(args.u, a_deg)
        rows.append([a_deg, fx, fy, clp, cdp])

    rows = np.array(rows)
    np.savetxt(
        os.path.join(args.outdir, "alpha_sweep_pressure_coeffs.csv"),
        rows,
        delimiter=",",
        header="alpha_deg,Fx,Fy,CL_pressure,CD_pressure",
        comments="",
    )

    plt.figure()
    plt.plot(rows[:, 0], rows[:, 3], label="CL pressure")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("alpha [deg]")
    plt.ylabel("CL_pressure")
    plt.title("Pressure lift coefficient vs alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cl_pressure_vs_alpha.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(rows[:, 0], rows[:, 4], label="CD pressure")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("alpha [deg]")
    plt.ylabel("CD_pressure")
    plt.title("Pressure drag coefficient vs alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cd_pressure_vs_alpha.png"), dpi=150)
    plt.close()

    return rows


def u_scaling_test(alpha_deg=5.0):
    Us = np.linspace(U_MIN, U_MAX, 9)
    rows = []

    for U in Us:
        fx, fy, clp, cdp = pressure_force(U, alpha_deg)
        rows.append([U, fx, fy, clp, cdp])

    rows = np.array(rows)
    np.savetxt(
        os.path.join(args.outdir, "u_scaling_pressure.csv"),
        rows,
        delimiter=",",
        header="U,Fx,Fy,CL_pressure,CD_pressure",
        comments="",
    )

    plt.figure()
    plt.plot(rows[:, 0], rows[:, 2], "o-", label="Fy")
    plt.plot(rows[:, 0], rows[0, 2] * (rows[:, 0] / rows[0, 0])**2, "--", label="ideal U^2 scaling from first point")
    plt.grid()
    plt.xlabel("U")
    plt.ylabel("pressure lift force Fy")
    plt.title(f"U scaling at alpha={alpha_deg} deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "u_scaling_force.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(rows[:, 0], rows[:, 3], "o-", label="CL_pressure")
    plt.grid()
    plt.xlabel("U")
    plt.ylabel("CL_pressure")
    plt.title(f"CL invariance vs U, alpha={alpha_deg} deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "u_scaling_cl.png"), dpi=150)
    plt.close()

    return rows


def plot_surface_pressure(alpha_deg=5.0, U_value=1.0):
    p = surface_pressure(U_value, alpha_deg)

    n = len(p)
    half = n // 2

    # construction du contour :
    # première moitié approx upper leading->trailing,
    # deuxième moitié lower trailing->leading.
    p_upper = p[:half]
    p_lower = p[half:][::-1]

    x_upper = SURFACE["xm"][:half]
    x_lower = SURFACE["xm"][half:][::-1]

    plt.figure(figsize=(9, 4))
    plt.plot(x_upper, p_upper, label="upper")
    plt.plot(x_lower, p_lower, label="lower")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("x along chord")
    plt.ylabel("p")
    plt.title(f"Surface pressure, U={U_value}, alpha={alpha_deg} deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"surface_pressure_alpha_{alpha_deg:+.1f}.png"), dpi=150)
    plt.close()

    dp = p_lower - p_upper
    plt.figure(figsize=(9, 4))
    plt.plot(x_upper, dp, label="p_lower - p_upper")
    plt.axhline(0, color="k", linewidth=0.8)
    plt.grid()
    plt.xlabel("x along chord")
    plt.ylabel("pressure difference")
    plt.title(f"Pressure jump, U={U_value}, alpha={alpha_deg} deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"surface_pressure_jump_alpha_{alpha_deg:+.1f}.png"), dpi=150)
    plt.close()


# ============================================================
# 4. Field and residual maps
# ============================================================

@torch.no_grad()
def plot_field(U_value=1.0, alpha_deg=5.0, nx_grid=160, ny_grid=80):
    alpha_value = math.radians(alpha_deg)

    xs = np.linspace(X_MIN, X_MAX, nx_grid, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, ny_grid, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    U = np.full((pts.shape[0], 1), U_value, dtype=np.float32)
    a = np.full((pts.shape[0], 1), alpha_value, dtype=np.float32)

    x = pts[:, 0:1].astype(np.float32)
    y = pts[:, 1:2].astype(np.float32)

    up, vp, pp = eval_model_np(x, y, U, a)

    u = up.reshape(ny_grid, nx_grid)
    v = vp.reshape(ny_grid, nx_grid)
    p = pp.reshape(ny_grid, nx_grid)

    mask = inside.reshape(ny_grid, nx_grid)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)
    p = np.where(mask, np.nan, p)

    plt.figure(figsize=(11, 4))
    plt.contourf(XX, YY, p, levels=60)
    plt.colorbar(label="p")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Pressure field, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "field_pressure.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 4))
    speed = np.sqrt(u**2 + v**2)
    plt.contourf(XX, YY, speed, levels=60)
    plt.colorbar(label="speed")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Speed field, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "field_speed.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 4))
    skip = (slice(None, None, 5), slice(None, None, 5))
    plt.quiver(XX[skip], YY[skip], u[skip], v[skip])
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Velocity field, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "field_velocity_quiver.png"), dpi=150)
    plt.close()


def residual_map(U_value=1.0, alpha_deg=5.0, nx_grid=120, ny_grid=60):
    alpha_value = math.radians(alpha_deg)

    xs = np.linspace(X_MIN, X_MAX, nx_grid, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, ny_grid, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    U = np.full((pts.shape[0], 1), U_value, dtype=np.float32)
    a = np.full((pts.shape[0], 1), alpha_value, dtype=np.float32)

    X = make_input_np(
        pts[:, 0:1].astype(np.float32),
        pts[:, 1:2].astype(np.float32),
        U,
        a,
        requires_grad=True,
    )

    mx, my, div = ns_residual(model, X)

    mx = mx.detach().cpu().numpy().reshape(ny_grid, nx_grid)
    my = my.detach().cpu().numpy().reshape(ny_grid, nx_grid)
    div = div.detach().cpu().numpy().reshape(ny_grid, nx_grid)

    res = np.sqrt(mx**2 + my**2)
    mask = inside.reshape(ny_grid, nx_grid)

    res = np.where(mask, np.nan, res)
    div = np.where(mask, np.nan, div)

    plt.figure(figsize=(11, 4))
    plt.contourf(XX, YY, np.log10(res + 1e-10), levels=60)
    plt.colorbar(label="log10(momentum residual)")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Momentum residual, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "residual_momentum_log.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.contourf(XX, YY, np.log10(np.abs(div) + 1e-10), levels=60)
    plt.colorbar(label="log10(abs(div))")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Divergence residual, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "residual_divergence_log.png"), dpi=150)
    plt.close()


# ============================================================
# Run all diagnostics
# ============================================================

plot_geometry()
metrics = validation_metrics(args.n_val)
sweep_rows = alpha_sweep()
u_rows = u_scaling_test(alpha_deg=args.alpha_deg)

for a_deg in [-10, -5, 0, 5, 10]:
    plot_surface_pressure(alpha_deg=a_deg, U_value=args.u)

plot_field(U_value=args.u, alpha_deg=args.alpha_deg)
residual_map(U_value=args.u, alpha_deg=args.alpha_deg)

# Summary text
summary_path = os.path.join(args.outdir, "diag_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== PINN Airfoil V1 diagnostics ===\n\n")
    f.write(f"checkpoint: {args.checkpoint}\n")
    f.write(f"device: {DEVICE}\n")
    f.write(f"domain: x=[{X_MIN},{X_MAX}], y=[{Y_MIN},{Y_MAX}]\n")
    f.write(f"U range: [{U_MIN},{U_MAX}]\n")
    f.write(f"alpha max deg: {math.degrees(ALPHA_MAX):.3f}\n")
    f.write(f"nu: {NU}\n")
    f.write(f"airfoil signed area: {SURFACE['area']:+.6e}\n")
    f.write(f"surface perimeter: {SURFACE['ds'].sum():.6e}\n\n")

    f.write("=== Validation metrics ===\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.6e}\n")

    f.write("\n=== Pressure force alpha sweep quick read ===\n")
    for target in [-10, -5, 0, 5, 10]:
        idx = np.argmin(np.abs(sweep_rows[:, 0] - target))
        row = sweep_rows[idx]
        f.write(
            f"alpha={row[0]:+.1f} deg | "
            f"Fx={row[1]:+.6e}, Fy={row[2]:+.6e}, "
            f"CLp={row[3]:+.6e}, CDp={row[4]:+.6e}\n"
        )

    f.write("\n=== U scaling quick read ===\n")
    for row in u_rows:
        f.write(
            f"U={row[0]:.3f} | "
            f"Fx={row[1]:+.6e}, Fy={row[2]:+.6e}, "
            f"CLp={row[3]:+.6e}, CDp={row[4]:+.6e}\n"
        )

print(f"Diagnostics written to: {args.outdir}")
print(f"Summary: {summary_path}")
for k, v in metrics.items():
    print(f"{k}: {v:.6e}")