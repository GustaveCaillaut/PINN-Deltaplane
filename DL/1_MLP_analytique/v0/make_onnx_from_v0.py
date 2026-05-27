import torch
import torch.nn as nn


# ------------------------------------------------------------
# Model definition
# MUST match the training architecture exactly
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


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load("aero_mlp.pt", map_location="cpu", weights_only=False)

model = AeroMLP()
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully")


# ------------------------------------------------------------
# Export ONNX
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

print("Exported aero_mlp.onnx")