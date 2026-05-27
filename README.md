# Neural Aerodynamics in Unreal Engine

This repository contains an Unreal Engine project and the associated Python experiments developed for a scientific project on neural aerodynamic modelling.

The goal of the project is to simulate aerodynamic lift, drag, and eventually aerodynamic moments in real time, while exploring different neural approaches for modelling the flow around an airfoil.

The project evolved from simple analytical aerodynamic laws, to direct neural regressors, to Physics-Informed Neural Networks (PINNs), and finally to a supervised surface-pressure surrogate based on XFOIL data.

---

## Releases

Compiled versions of the project are available in the GitHub **Releases** section.

The release zip contains the packaged Unreal project, including the executable `.exe`.

After downloading and extracting the zip, run the `.exe` from inside the extracted folder. The executable should not be moved outside its packaged directory, because Unreal builds depend on associated `.pak` and content files.

---

## Repository structure

```text
.
├── SMLEProject/
│   ├── Content/
│   ├── Source/
│   ├── Config/
│   └── ...
│
├── DL/
│   ├── PINN experiments
│   ├── XFOIL dataset generation
│   ├── supervised Cp surrogate training
│   ├── plots and experiment results
│   └── notebooks / scripts
│
└── README.md
```

---

## Unreal Engine project

The Unreal Engine project contains the interactive simulation.

It includes:

- the Unreal level and assets;
- the player object and physics setup;
- Blueprint logic for computing speed, angle of attack, lift and drag;
- C++ components for neural-network inference;
- integration of ONNX models through Unreal's NNE / ONNX Runtime;
- aerodynamic force application in real time.

The project contains both visual assets and the physics/aerodynamics implementation.

The final version uses a neural surrogate that predicts the pressure coefficient `Cp` on the airfoil surface. The pressure is then integrated to obtain lift, while drag is currently handled with a simple analytical model.

---

## C++ code and Blueprints

The C++ code is located in the Unreal project `Source/` directory.

Important elements include:

- neural network loading and inference;
- surface discretization of the NACA airfoil;
- batch evaluation of the ONNX model;
- pressure integration on the airfoil surface;
- computation of aerodynamic coefficients.

Blueprints are used for:

- player movement and physics;
- computation of velocity and angle of attack;
- conversion of `CL` and `CD` into Unreal forces;
- debug visualization and simulation integration.

---

## Deep learning experiments

The `DL/` folder contains the Python side of the project.

It includes:

- early analytical and MLP tests;
- PINN implementations for incompressible Navier--Stokes;
- tests with different boundary conditions;
- Kutta-condition experiments;
- Reynolds-number sweeps;
- OpenBC experiments discussed during the project;
- XFOIL data generation scripts;
- supervised and hybrid data+PDE experiments;
- final supervised `Cp` surrogate training;
- plots, logs, summaries, and experiment results.

The PINN experiments are included even though they were not used as the final Unreal model, because a significant part of the project consisted in understanding why a pure PINN formulation did not recover a realistic lifting solution.

The main conclusion from these experiments is that the pure PINN tended to converge to almost non-lifting solutions, even when the wall condition and PDE residuals were reasonably satisfied. Adding surface-pressure data from XFOIL made the model recover realistic pressure distributions and lift coefficients.

---

## Final neural model

The final approach is a supervised surface-pressure surrogate.

The model takes as input:

```text
[x, y, side, sin(alpha), cos(alpha)]
```

where:

- `x, y` are airfoil surface coordinates;
- `side` indicates upper or lower surface;
- `alpha` is the angle of attack.

The model outputs:

```text
Cp
```

The pressure coefficient is then integrated over the airfoil surface:

```text
F = - ∫ Cp n ds
```

This gives a physically interpretable lift coefficient while remaining simple enough for real-time inference in Unreal.

---

## Notes and limitations

The final model should be interpreted as a surrogate of XFOIL data, not as a full CFD solver.

Current limitations include:

- validity limited to the angle-of-attack range used during training;
- no reliable post-stall model yet;
- drag is still handled analytically rather than fully predicted from local shear stress;
- the PINN volume fields are diagnostic and should not be interpreted as fully accurate CFD solutions;
- the model is currently tied to a NACA 2412 airfoil.

Possible future improvements include:

- adding a post-stall model;
- training on higher-fidelity CFD data;
- predicting local traction instead of only pressure;
- adding Reynolds-number dependence;
- extending the model to variable airfoil geometries;
- improving aerodynamic moment modelling.

---

## Project status

The current repository represents the final state of the course project.

The Unreal simulation is functional, the neural model is integrated, and the Python experiments document the progression from PINNs to the final supervised pressure-surrogate approach.
