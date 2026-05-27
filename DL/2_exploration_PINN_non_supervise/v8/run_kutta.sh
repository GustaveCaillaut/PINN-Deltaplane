#!/usr/bin/env bash

set -u

PYTHON=${PYTHON:-python}

SCRIPT="train_pinn_airfoil_v8_alpha_data_kutta.py"
XFOIL_GEN="generate_xfoil_cp_dataset.py"

LOGDIR="logs_v8_short"
mkdir -p "$LOGDIR"

ALPHAS="-15,-10,-5,0,5,10,15"

CPDATA_200K="xfoil_cp_dataset_naca2412_full/xfoil_cp_dataset.csv"
CPDATA_20K="xfoil_cp_dataset_naca2412_Re20k/xfoil_cp_dataset.csv"

DIAG_ARGS=(
  --diag-alphas=-10,-5,0,5,10,15
  --sweep-alphas=-15,-10,-5,0,5,10,15
)

COMMON_SHORT=(
  --adam-steps 8000
  --lbfgs-steps 100
  --warmup-steps 300
  --lr 2e-4
  --n-f 2048
  --n-near 2048
  --n-boundary 1024
  --n-wall 512
  --n-data 2048
  --n-kutta 256
  --print-every 250
  --lbfgs-print-every 25
  --fourier-mode multiscale
  --fourier-n-freqs 16
  --fourier-scale 1.0
)

COMMON_DEFAULT_PHYSICS=(
  --w-pde 1
  --w-div 5
  --w-near-pde 10
  --w-near-div 20
  --w-wall 50
  --w-inflow-vel 10
  --w-pressure-gauge 1
  --w-outflow-cp 0
  --w-far-cp 0
)

COMMON_WEAK_PDE=(
  --w-pde 0.1
  --w-div 1
  --w-near-pde 1
  --w-near-div 5
  --w-wall 10
  --w-inflow-vel 10
  --w-pressure-gauge 1
  --w-outflow-cp 0
  --w-far-cp 0
)

run_and_log() {
  local name="$1"
  shift

  echo ""
  echo "============================================================"
  echo "RUNNING: $name"
  echo "============================================================"
  echo ""

  "$@" 2>&1 | tee "$LOGDIR/${name}.log"

  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "WARNING: run '$name' failed with status $status"
    echo "Continuing..."
  fi
}

run_and_log "v8_short_00_alpha_kutta_only" \
  "$PYTHON" -u "$SCRIPT" \
    --outdir runs_v8_short_00_alpha_kutta_only \
    --U 2 \
    --nu 0.01 \
    --alpha-min-deg -15 \
    --alpha-max-deg 15 \
    --alpha-sampling range \
    --w-data 0 \
    --w-kutta 1 \
    --w-kutta-tangent 1 \
    --w-wake-recovery 0 \
    "${COMMON_SHORT[@]}" \
    "${COMMON_DEFAULT_PHYSICS[@]}" \
    "${DIAG_ARGS[@]}"