@echo off
setlocal

REM ============================================================
REM PINN Airfoil V3 fixed-case batch experiments
REM ============================================================

set SCRIPT=train_pinn_airfoil_v3_fixed.py

REM ------------------------------------------------------------
REM Run 1 : baseline
REM Re = 50
REM ------------------------------------------------------------
echo.
echo ============================================================
echo RUN 1 - baseline U=1 nu=0.02
echo ============================================================

python %SCRIPT% ^
  --outdir runs_v3_u1_nu002 ^
  --U 1 ^
  --nu 0.02 ^
  --w-cl 0 ^
  --adam-steps 6000 ^
  --lbfgs-steps 0

IF ERRORLEVEL 1 goto :error

REM ------------------------------------------------------------
REM Run 2 : higher U
REM Re = 100
REM ------------------------------------------------------------
echo.
echo ============================================================
echo RUN 2 - U=2 nu=0.02
echo ============================================================

python %SCRIPT% ^
  --outdir runs_v3_u2_nu002 ^
  --U 2 ^
  --nu 0.02 ^
  --w-cl 0 ^
  --adam-steps 6000 ^
  --lbfgs-steps 0

IF ERRORLEVEL 1 goto :error

REM ------------------------------------------------------------
REM Run 3 : even higher U
REM Re = 150
REM ------------------------------------------------------------
echo.
echo ============================================================
echo RUN 3 - U=3 nu=0.02
echo ============================================================

python %SCRIPT% ^
  --outdir runs_v3_u3_nu002 ^
  --U 3 ^
  --nu 0.02 ^
  --w-cl 0 ^
  --adam-steps 6000 ^
  --lbfgs-steps 0

IF ERRORLEVEL 1 goto :error

REM ------------------------------------------------------------
REM Run 4 : lower viscosity
REM Re = 200
REM ------------------------------------------------------------
echo.
echo ============================================================
echo RUN 4 - U=2 nu=0.01
echo ============================================================

python %SCRIPT% ^
  --outdir runs_v3_u2_nu001 ^
  --U 2 ^
  --nu 0.01 ^
  --w-cl 0 ^
  --adam-steps 6000 ^
  --lbfgs-steps 0

IF ERRORLEVEL 1 goto :error

REM ------------------------------------------------------------
REM Run 5 : even lower viscosity
REM Re = 400
REM ------------------------------------------------------------
echo.
echo ============================================================
echo RUN 5 - U=2 nu=0.005
echo ============================================================

python %SCRIPT% ^
  --outdir runs_v3_u2_nu0005 ^
  --U 2 ^
  --nu 0.005 ^
  --w-cl 0 ^
  --adam-steps 6000 ^
  --lbfgs-steps 0

IF ERRORLEVEL 1 goto :error

echo.
echo ============================================================
echo ALL RUNS COMPLETED
echo ============================================================

goto :end

:error
echo.
echo ============================================================
echo ERROR DETECTED - STOPPING BATCH
echo ============================================================

:end
pause