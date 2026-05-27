import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Analytical teacher model
# ------------------------------------------------------------

def aero_coefficients(alpha):
    """
    alpha in radians.
    Returns CL, CD.
    """
    CL_slope = 4.5
    CL_max = 1.2
    CD0 = 0.08
    k = 0.08
    CD_flat = 0.8

    CL = CL_max * torch.tanh((CL_slope * alpha) / CL_max)
    CD = CD0 + k * CL**2 + CD_flat * torch.sin(alpha)**2
    return CL, CD


# ------------------------------------------------------------
# 2. Dataset
# ------------------------------------------------------------

alpha_min = np.deg2rad(-80)
alpha_max = np.deg2rad(80)

n_samples = 5000
alpha = torch.linspace(alpha_min, alpha_max, n_samples).unsqueeze(1)

CL, CD = aero_coefficients(alpha)
y = torch.cat([CL, CD], dim=1)

# Normalize input alpha roughly to [-1, 1]
alpha_scale = max(abs(alpha_min), abs(alpha_max))
x = alpha / alpha_scale


# ------------------------------------------------------------
# 3. MLP
# ------------------------------------------------------------

class AeroMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


model = AeroMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# ------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------

for epoch in range(3000):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"epoch {epoch:4d} | loss = {loss.item():.6e}")


# ------------------------------------------------------------
# 5. Plot check
# ------------------------------------------------------------

with torch.no_grad():
    pred = model(x)
    pred_CL = pred[:, 0]
    pred_CD = pred[:, 1]

alpha_deg = alpha.squeeze().numpy() * 180 / np.pi

plt.figure()
plt.plot(alpha_deg, CL.squeeze().numpy(), label="CL teacher")
plt.plot(alpha_deg, pred_CL.numpy(), "--", label="CL MLP")
plt.legend()
plt.xlabel("alpha [deg]")
plt.ylabel("CL")
plt.grid()
plt.show()

plt.figure()
plt.plot(alpha_deg, CD.squeeze().numpy(), label="CD teacher")
plt.plot(alpha_deg, pred_CD.numpy(), "--", label="CD MLP")
plt.legend()
plt.xlabel("alpha [deg]")
plt.ylabel("CD")
plt.grid()
plt.show()


# ------------------------------------------------------------
# 6. Save PyTorch model
# ------------------------------------------------------------

torch.save({
    "model_state_dict": model.state_dict(),
    "alpha_scale": alpha_scale,
}, "aero_mlp.pt")

print("Saved aero_mlp.pt")


# ------------------------------------------------------------
# 7. Export ONNX
# ------------------------------------------------------------

dummy_input = torch.zeros(1, 1)

torch.onnx.export(
    model,
    dummy_input,
    "aero_mlp.onnx",
    input_names=["alpha_normalized"],
    output_names=["CL_CD"],
    dynamic_axes={
        "alpha_normalized": {0: "batch"},
        "CL_CD": {0: "batch"},
    },
    opset_version=17,
)

print("Saved aero_mlp.onnx")