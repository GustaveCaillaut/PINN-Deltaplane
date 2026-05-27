import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.path import Path


# ============================================================
# 0. Config
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

# Domaine fluide, en unités de corde.
# La corde de l'aile vaut 1.
X_MIN, X_MAX = -1.5, 2.5
Y_MIN, Y_MAX = -1.0, 1.0

# Paramètres du flux.
# U est ici adimensionné / normalisé.
U_MIN, U_MAX = 0.5, 1.5
ALPHA_MAX_DEG = 15.0
ALPHA_MAX = math.radians(ALPHA_MAX_DEG)

# Viscosité cinématique adimensionnée.
# Corde = 1, donc Re ~ U / nu.
# Avec nu = 0.02 et U ~ 1, Re ~ 50.
NU = 0.02

# Tailles de batch
N_F = 4096        # points collocation fluide
N_INLET = 512
N_FAR = 512
N_OUTLET = 512
N_WALL = 512

# Training
N_STEPS = 8000
LR = 1e-3
PRINT_EVERY = 250

# Loss weights
W_PDE = 1.0
W_DIV = 1.0
W_INLET = 10.0
W_FAR = 5.0
W_OUTLET = 1.0
W_WALL = 20.0


# ============================================================
# 1. Géométrie NACA 2412
# ============================================================

def naca2412_points(n=300):
    """
    Retourne le contour fermé d'un profil NACA 2412.
    Corde = 1.
    Bord d'attaque vers x=0, bord de fuite vers x=1.
    Puis on recentre autour de x=0 : x in [-0.5, 0.5].
    """
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

    # Contour fermé :
    # upper: leading edge -> trailing edge
    # lower: trailing edge -> leading edge
    X = np.concatenate([xu, xl[::-1]])
    Y = np.concatenate([yu, yl[::-1]])

    X = X - 0.5

    return X.astype(np.float32), Y.astype(np.float32)


def polygon_signed_area(x, y):
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * np.sum(x * y2 - x2 * y)


def build_surface_geometry(x, y):
    """
    Construit les milieux de segments, longueurs ds, normales extérieures.
    """
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    dx = x_next - x
    dy = y_next - y

    ds = np.sqrt(dx**2 + dy**2) + 1e-12

    xm = 0.5 * (x + x_next)
    ym = 0.5 * (y + y_next)

    area = polygon_signed_area(x, y)

    # Si contour horaire : normale extérieure = normale gauche (-dy, dx)
    # Si contour antihoraire : normale extérieure = normale droite (dy, -dx)
    if area < 0:
        nx = -dy / ds
        ny = dx / ds
    else:
        nx = dy / ds
        ny = -dx / ds

    surface = {
        "x": x,
        "y": y,
        "xm": xm.astype(np.float32),
        "ym": ym.astype(np.float32),
        "nx": nx.astype(np.float32),
        "ny": ny.astype(np.float32),
        "ds": ds.astype(np.float32),
        "area": area,
    }
    return surface


AIRFOIL_X, AIRFOIL_Y = naca2412_points(n=300)
AIRFOIL_PATH = Path(np.stack([AIRFOIL_X, AIRFOIL_Y], axis=1))
SURFACE = build_surface_geometry(AIRFOIL_X, AIRFOIL_Y)


# ============================================================
# 2. Sampling
# ============================================================

def sample_params(n):
    U = np.random.uniform(U_MIN, U_MAX, size=(n, 1)).astype(np.float32)
    alpha = np.random.uniform(-ALPHA_MAX, ALPHA_MAX, size=(n, 1)).astype(np.float32)
    return U, alpha


def sample_fluid_points(n):
    """
    Sample dans le rectangle, puis rejette les points à l'intérieur de l'aile.
    """
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


def sample_inlet(n):
    x = np.full((n, 1), X_MIN, dtype=np.float32)
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    return x, y


def sample_outlet(n):
    x = np.full((n, 1), X_MAX, dtype=np.float32)
    y = np.random.uniform(Y_MIN, Y_MAX, size=(n, 1)).astype(np.float32)
    return x, y


def sample_farfield(n):
    """
    Top/bottom boundaries.
    """
    n_top = n // 2
    n_bot = n - n_top

    x_top = np.random.uniform(X_MIN, X_MAX, size=(n_top, 1)).astype(np.float32)
    y_top = np.full((n_top, 1), Y_MAX, dtype=np.float32)

    x_bot = np.random.uniform(X_MIN, X_MAX, size=(n_bot, 1)).astype(np.float32)
    y_bot = np.full((n_bot, 1), Y_MIN, dtype=np.float32)

    x = np.concatenate([x_top, x_bot], axis=0)
    y = np.concatenate([y_top, y_bot], axis=0)

    return x, y


def sample_wall(n):
    """
    Sample des points sur la surface de l'aile.
    """
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


# ============================================================
# 3. Modèle MLP PINN
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
        in_dim = 4
        out_dim = 3

        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_dim))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inp):
        # inp = [x, y, U, alpha]
        z = (inp - self.center) / self.scale
        return self.net(z)


model = PINNAirfoil(width=128, depth=7).to(DEVICE)


# ============================================================
# 4. Autograd utils
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
    """
    X shape: [N,4] = x,y,U,alpha
    returns residuals:
    momentum_x, momentum_y, divergence
    """
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

    # Steady incompressible Navier-Stokes:
    # u u_x + v u_y = -p_x + nu Δu
    # u v_x + v v_y = -p_y + nu Δv
    mom_x = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    mom_y = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

    div = u_x + v_y

    return mom_x, mom_y, div


def freestream(U, alpha):
    """
    Convention V1 :
    flux gauche -> droite.
    alpha > 0 => flux légèrement montant.
    """
    u_inf = U * torch.cos(alpha)
    v_inf = U * torch.sin(alpha)
    return u_inf, v_inf


# ============================================================
# 5. Training
# ============================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
history = []

for step in range(1, N_STEPS + 1):
    optimizer.zero_grad()

    # -----------------------------
    # PDE collocation points
    # -----------------------------
    xf, yf = sample_fluid_points(N_F)
    Uf, af = sample_params(N_F)
    Xf = make_input(xf, yf, Uf, af, requires_grad=True)

    mom_x, mom_y, div = ns_residual(model, Xf)

    loss_pde = torch.mean(mom_x**2) + torch.mean(mom_y**2)
    loss_div = torch.mean(div**2)

    # -----------------------------
    # Inlet BC
    # -----------------------------
    xi, yi = sample_inlet(N_INLET)
    Ui, ai = sample_params(N_INLET)
    Xi = make_input(xi, yi, Ui, ai)

    out_i = model(Xi)
    ui = out_i[:, 0:1]
    vi = out_i[:, 1:2]

    Ui_t = Xi[:, 2:3]
    ai_t = Xi[:, 3:4]
    ui_target, vi_target = freestream(Ui_t, ai_t)

    loss_inlet = torch.mean((ui - ui_target)**2 + (vi - vi_target)**2)

    # -----------------------------
    # Farfield top/bottom BC
    # -----------------------------
    xb, yb = sample_farfield(N_FAR)
    Ub, ab = sample_params(N_FAR)
    Xb = make_input(xb, yb, Ub, ab)

    out_b = model(Xb)
    ub = out_b[:, 0:1]
    vb = out_b[:, 1:2]

    Ub_t = Xb[:, 2:3]
    ab_t = Xb[:, 3:4]
    ub_target, vb_target = freestream(Ub_t, ab_t)

    loss_far = torch.mean((ub - ub_target)**2 + (vb - vb_target)**2)

    # -----------------------------
    # Outlet pressure reference
    # -----------------------------
    xo, yo = sample_outlet(N_OUTLET)
    Uo, ao = sample_params(N_OUTLET)
    Xo = make_input(xo, yo, Uo, ao)

    out_o = model(Xo)
    po = out_o[:, 2:3]

    loss_outlet = torch.mean(po**2)

    # -----------------------------
    # Wall no-slip
    # -----------------------------
    xw, yw = sample_wall(N_WALL)
    Uw, aw = sample_params(N_WALL)
    Xw = make_input(xw, yw, Uw, aw)

    out_w = model(Xw)
    uw = out_w[:, 0:1]
    vw = out_w[:, 1:2]

    loss_wall = torch.mean(uw**2 + vw**2)

    # -----------------------------
    # Total
    # -----------------------------
    loss = (
        W_PDE * loss_pde
        + W_DIV * loss_div
        + W_INLET * loss_inlet
        + W_FAR * loss_far
        + W_OUTLET * loss_outlet
        + W_WALL * loss_wall
    )

    loss.backward()
    optimizer.step()

    history.append([
        loss.item(),
        loss_pde.item(),
        loss_div.item(),
        loss_inlet.item(),
        loss_far.item(),
        loss_outlet.item(),
        loss_wall.item(),
    ])

    if step % PRINT_EVERY == 0 or step == 1:
        print(
            f"[{step:6d}/{N_STEPS}] "
            f"loss={loss.item():.3e} | "
            f"pde={loss_pde.item():.3e} | "
            f"div={loss_div.item():.3e} | "
            f"in={loss_inlet.item():.3e} | "
            f"far={loss_far.item():.3e} | "
            f"out={loss_outlet.item():.3e} | "
            f"wall={loss_wall.item():.3e}"
        )


# ============================================================
# 6. Sauvegarde
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "x_min": X_MIN,
            "x_max": X_MAX,
            "y_min": Y_MIN,
            "y_max": Y_MAX,
            "u_min": U_MIN,
            "u_max": U_MAX,
            "alpha_max": ALPHA_MAX,
            "nu": NU,
        },
    },
    "pinn_airfoil_v1.pt",
)

print("Saved pinn_airfoil_v1.pt")


# ============================================================
# 7. Export ONNX forward only
# ============================================================

model.eval()

dummy = torch.zeros(1, 4, device=DEVICE, dtype=DTYPE)

torch.onnx.export(
    model,
    dummy,
    "pinn_airfoil_v1.onnx",
    input_names=["xy_U_alpha"],
    output_names=["uvp"],
    dynamic_axes={
        "xy_U_alpha": {0: "batch"},
        "uvp": {0: "batch"},
    },
    opset_version=17,
)

print("Saved pinn_airfoil_v1.onnx")


# ============================================================
# 8. Diagnostics : intégration pression
# ============================================================

@torch.no_grad()
def pressure_force(model, U_value=1.0, alpha_deg=5.0):
    """
    Force pression sur l'aile :
        F = - ∫ p n ds

    Convention :
    - n = normale extérieure solide -> fluide
    - p = pression prédite
    - flux moyen gauche -> droite
    - Fx positif = force vers la droite
    - Fy positif = portance vers le haut
    """
    alpha_value = math.radians(alpha_deg)

    x = SURFACE["xm"].reshape(-1, 1)
    y = SURFACE["ym"].reshape(-1, 1)
    nx = SURFACE["nx"].reshape(-1, 1)
    ny = SURFACE["ny"].reshape(-1, 1)
    ds = SURFACE["ds"].reshape(-1, 1)

    U = np.full_like(x, U_value, dtype=np.float32)
    a = np.full_like(x, alpha_value, dtype=np.float32)

    X = make_input(x, y, U, a, requires_grad=False)

    out = model(X)
    p = out[:, 2:3].cpu().numpy()

    fx = -np.sum(p * nx * ds)
    fy = -np.sum(p * ny * ds)

    q = 0.5 * U_value**2
    c_ref = 1.0

    cd_pressure = fx / (q * c_ref)
    cl_pressure = fy / (q * c_ref)

    return fx, fy, cl_pressure, cd_pressure


for a_deg in [-10, -5, 0, 5, 10]:
    fx, fy, clp, cdp = pressure_force(model, U_value=1.0, alpha_deg=a_deg)
    print(
        f"alpha={a_deg:+.1f} deg | "
        f"Fx={fx:+.4e}, Fy={fy:+.4e}, "
        f"CL_pressure={clp:+.4e}, CD_pressure={cdp:+.4e}"
    )


# ============================================================
# 9. Plots rapides
# ============================================================

history = np.array(history)

plt.figure()
plt.semilogy(history[:, 0], label="total")
plt.semilogy(history[:, 1], label="pde")
plt.semilogy(history[:, 2], label="div")
plt.semilogy(history[:, 3], label="inlet")
plt.semilogy(history[:, 4], label="far")
plt.semilogy(history[:, 5], label="outlet")
plt.semilogy(history[:, 6], label="wall")
plt.legend()
plt.grid()
plt.xlabel("step")
plt.ylabel("loss")
plt.title("PINN losses")
plt.tight_layout()
plt.savefig("pinn_airfoil_v1_losses.png", dpi=150)
plt.show()


@torch.no_grad()
def plot_field(model, U_value=1.0, alpha_deg=5.0, nx_grid=140, ny_grid=70):
    alpha_value = math.radians(alpha_deg)

    xs = np.linspace(X_MIN, X_MAX, nx_grid, dtype=np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, ny_grid, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    inside = AIRFOIL_PATH.contains_points(pts)

    U = np.full((pts.shape[0], 1), U_value, dtype=np.float32)
    a = np.full((pts.shape[0], 1), alpha_value, dtype=np.float32)

    Xinp = np.concatenate(
        [pts[:, 0:1], pts[:, 1:2], U, a],
        axis=1,
    ).astype(np.float32)

    Xtorch = torch.tensor(Xinp, device=DEVICE, dtype=DTYPE)
    out = model(Xtorch).cpu().numpy()

    u = out[:, 0].reshape(ny_grid, nx_grid)
    v = out[:, 1].reshape(ny_grid, nx_grid)
    p = out[:, 2].reshape(ny_grid, nx_grid)

    mask = inside.reshape(ny_grid, nx_grid)

    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)
    p = np.where(mask, np.nan, p)

    plt.figure(figsize=(10, 4))
    plt.contourf(XX, YY, p, levels=50)
    plt.colorbar(label="p")
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Pressure field, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig("pinn_airfoil_v1_pressure.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 4))
    skip = (slice(None, None, 5), slice(None, None, 5))
    plt.quiver(XX[skip], YY[skip], u[skip], v[skip])
    plt.plot(AIRFOIL_X, AIRFOIL_Y, "k", linewidth=2)
    plt.axis("equal")
    plt.title(f"Velocity field, U={U_value}, alpha={alpha_deg} deg")
    plt.tight_layout()
    plt.savefig("pinn_airfoil_v1_velocity.png", dpi=150)
    plt.show()


plot_field(model, U_value=1.0, alpha_deg=5.0)