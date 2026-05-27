#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
PRESET=${PRESET:-full}
OUTDIR=${OUTDIR:-runs_pinn_failure_v2_full}
CPDATA=${CPDATA:-}

ARGS=(
  --preset "$PRESET"
  --outdir "$OUTDIR"
)

if [ -n "$CPDATA" ]; then
  ARGS+=(--cp-data-csv "$CPDATA")
fi

mkdir -p logs_pinn_failure_v2
LOG="logs_pinn_failure_v2/${OUTDIR}.log"

echo "Running PINN failure suite"
echo "Preset: $PRESET"
echo "Outdir: $OUTDIR"
echo "CPDATA: ${CPDATA:-<none>}"
echo "Log: $LOG"

$PYTHON -u run_pinn_failure_suite_v2.py "${ARGS[@]}" 2>&1 | tee "$LOG"
