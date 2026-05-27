#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
PRESET=${PRESET:-full}
OUTDIR=${OUTDIR:-runs_pinn_openbc_v3_full}
CPDATA=${CPDATA:-}

ARGS=(
  --preset "$PRESET"
  --outdir "$OUTDIR"
)

if [ -n "$CPDATA" ]; then
  ARGS+=(--cp-data-csv "$CPDATA")
fi

mkdir -p logs_pinn_openbc_v3
LOG="logs_pinn_openbc_v3/${OUTDIR}.log"

echo "Running PINN open-boundary suite"
echo "Preset: $PRESET"
echo "Outdir: $OUTDIR"
echo "CPDATA: ${CPDATA:-<none>}"
echo "Log: $LOG"

$PYTHON -u run_pinn_openbc_suite_v3.py "${ARGS[@]}" 2>&1 | tee "$LOG"
