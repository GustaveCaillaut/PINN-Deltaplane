# PINN Open-Boundary V3

This version implements the corrected boundary-condition idea discussed with the professor.
It is meant to be a direct continuation of the previous PINN experiments, but with a cleaner far-field formulation.

## Files

- `pinn_airfoil_openbc_lab_v3.py`  
  Main library: geometry, sampling, PINN model, Navier--Stokes residuals, open-boundary losses, Kutta term, optional XFOIL Cp supervision, diagnostics and plots.

- `run_pinn_openbc_suite_v3.py`  
  Experiment runner. It launches the unsupervised open-boundary PINN and, if a Cp CSV is provided, the mixed data+PDE versions.

- `run_pinn_openbc_suite_v3.sh`  
  Shell launcher for the runner.

## Boundary conditions in this version

Let

\[
\mathbf U_\infty = U(\cos\alpha, \sin\alpha)
\]

and let \(\mathbf n\) be the outward normal of a rectangular outer boundary.

The right boundary is special:

\[
p = 0 \quad \text{on } x=x_{\max}.
\]

No velocity condition is imposed on the right boundary.

On the other external boundaries, compute

\[
s = \widehat{\mathbf U}_\infty \cdot \mathbf n.
\]

Then:

- if \(s < -\varepsilon\): inflow, impose full velocity

\[
u = U\cos\alpha, \qquad v = U\sin\alpha.
\]

- if \(|s| \le \varepsilon\): nearly parallel side boundary, impose no normal flux

\[
\mathbf u\cdot\mathbf n = 0.
\]

- if \(s > \varepsilon\): outgoing non-right boundary, impose only the normal component

\[
\mathbf u\cdot\mathbf n = \mathbf U_\infty\cdot\mathbf n.
\]

This avoids the older over-constrained condition where the full freestream velocity was imposed on all four boundaries.

## Main cases

### `30_openBC_unsupervised_Re200`

Pure unsupervised PINN:

- PDE residuals in the fluid domain;
- boosted PDE residuals near the airfoil;
- no-slip on the airfoil;
- corrected open-boundary conditions;
- right pressure condition;
- no XFOIL data;
- no Kutta.

### `30_openBC_unsupervised_kutta_Re200`

Same as above, but with the local wake-tangency Kutta penalty.

### `30_openBC_unsupervised_Re1000`

Same as the pure open-boundary run, but with larger Reynolds number.
Included in `full` and `overnight`, not in `smoke`.

### `30_openBC_unsupervised_kutta_Re1000`

Same as above, with Kutta.

### `31_openBC_xfoilData_weakPDE_Re200`

Mixed data+PDE version:

- XFOIL surface Cp supervision;
- weak PDE residual;
- weak boundary and wall terms;
- corrected open boundaries;
- no Kutta.

The goal is to show whether the network can recover a lifting pressure distribution when guided by surface pressure data.

### `31_openBC_xfoilData_weakPDE_kutta_Re200`

Same as the mixed version, but with the Kutta wake-tangency term.

### `31_openBC_xfoilData_strongerPDE_Re200`

A stronger PDE regularization variant. This is useful to see whether the PDE terms conflict with the XFOIL surface data.
Included in `full` and `overnight`, not in `smoke`.

## Commands

Smoke test:

```bash
python -u run_pinn_openbc_suite_v3.py \
  --preset smoke \
  --outdir runs_pinn_openbc_v3_smoke \
  --no-save-model
```

Full run without XFOIL data:

```bash
python -u run_pinn_openbc_suite_v3.py \
  --preset full \
  --outdir runs_pinn_openbc_v3_full
```

Full run with mixed data+PDE cases:

```bash
python -u run_pinn_openbc_suite_v3.py \
  --preset full \
  --outdir runs_pinn_openbc_v3_full_data \
  --cp-data-csv xfoil_cp_dataset_naca2412_dense_Re200k/xfoil_cp_dataset.csv
```

Shell launcher:

```bash
PRESET=full \
OUTDIR=runs_pinn_openbc_v3_full_data \
CPDATA=xfoil_cp_dataset_naca2412_dense_Re200k/xfoil_cp_dataset.csv \
./run_pinn_openbc_suite_v3.sh
```
