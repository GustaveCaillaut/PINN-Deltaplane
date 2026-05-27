@echo off
setlocal

set SCRIPT=train_pinn_airfoil_v6_fourier.py

echo ============================================================
echo V6 TEST 1 - no Fourier, only curriculum/loss changes
echo ============================================================
python %SCRIPT% ^
  --outdir runs_v6_01_no_fourier ^
  --fourier-mode none ^
  --adam-steps 20000 ^
  --lbfgs-steps 300 ^
  --warmup-steps 500 ^
  --lr 5e-4 ^
  --w-near-pde 20 ^
  --w-near-div 50 ^
  --w-cl 0
IF ERRORLEVEL 1 goto :error

echo ============================================================
echo V6 TEST 2 - multiscale Fourier, 8 freqs
echo ============================================================
python %SCRIPT% ^
  --outdir runs_v6_02_fourier_multi8 ^
  --fourier-mode multiscale ^
  --fourier-n-freqs 8 ^
  --fourier-scale 1.0 ^
  --adam-steps 20000 ^
  --lbfgs-steps 300 ^
  --warmup-steps 500 ^
  --lr 5e-4 ^
  --w-near-pde 20 ^
  --w-near-div 50 ^
  --w-cl 0
IF ERRORLEVEL 1 goto :error

echo ============================================================
echo V6 TEST 3 - multiscale Fourier, 16 freqs
echo ============================================================
python %SCRIPT% ^
  --outdir runs_v6_03_fourier_multi16 ^
  --fourier-mode multiscale ^
  --fourier-n-freqs 16 ^
  --fourier-scale 1.0 ^
  --adam-steps 20000 ^
  --lbfgs-steps 300 ^
  --warmup-steps 500 ^
  --lr 5e-4 ^
  --w-near-pde 20 ^
  --w-near-div 50 ^
  --w-cl 0
IF ERRORLEVEL 1 goto :error

echo ============================================================
echo V6 TEST 4 - random Fourier, moderate scale
echo ============================================================
python %SCRIPT% ^
  --outdir runs_v6_04_fourier_random16_s1 ^
  --fourier-mode random ^
  --fourier-n-freqs 16 ^
  --fourier-scale 1.0 ^
  --adam-steps 20000 ^
  --lbfgs-steps 300 ^
  --warmup-steps 500 ^
  --lr 5e-4 ^
  --w-near-pde 20 ^
  --w-near-div 50 ^
  --w-cl 0
IF ERRORLEVEL 1 goto :error

echo ============================================================
echo V6 TEST 5 - multiscale Fourier + slightly higher Re
echo ============================================================
python %SCRIPT% ^
  --outdir runs_v6_05_fourier_multi16_Re400 ^
  --fourier-mode multiscale ^
  --fourier-n-freqs 16 ^
  --fourier-scale 1.0 ^
  --U 2 ^
  --nu 0.005 ^
  --adam-steps 20000 ^
  --lbfgs-steps 300 ^
  --warmup-steps 500 ^
  --lr 5e-4 ^
  --w-near-pde 20 ^
  --w-near-div 50 ^
  --w-cl 0
IF ERRORLEVEL 1 goto :error

echo ============================================================
echo ALL V6 RUNS COMPLETED
echo ============================================================
goto :end

:error
echo ============================================================
echo ERROR DETECTED - STOPPING
echo ============================================================

:end
pause