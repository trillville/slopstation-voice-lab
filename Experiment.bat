@echo off
REM Seed-swept A/B: does --pad-then-mix beat the current data? One command,
REM walk away, ~3.5 h. Nothing here needs attention until it finishes.
REM
REM     Experiment.bat
REM
REM WHY THREE SEEDS PER ARM, and why this script exists at all: livekit pins no
REM RNG, so every run is a random draw, and two models trained on BYTE-IDENTICAL
REM features with the same recipe scored clean 64%% vs 86%% and +10 dB recall
REM 14%% vs 37%% (2026-08-17). Seed variance is larger than every effect this
REM project has chased, which is why single-run conclusions have had to be
REM retracted twice. An arm is a DISTRIBUTION over seeds; one model is an
REM anecdote. Compare the two ranges, never best-vs-best.
REM
REM ORDERING IS LOAD-BEARING. Both arms share one output directory, so arm A
REM must finish training before arm B's augment overwrites the features. Each
REM arm rebuilds once (--from augment) and its later seeds reuse that data
REM (--from train). Do not reorder.
REM
REM Artifacts and result keys carry the seed and the tag, so nothing collides:
REM   arm A  medium@s0    hey_alfred_medium_s0_v1.2.onnx
REM   arm B  medium@pad-s0  hey_alfred_medium_pad_s0_v1.2.onnx
setlocal
set TRAIN=%~dp0Train.bat
set BENCH=%~dp0Bench.bat
set FAILS=0

echo.
echo ==========================================================
echo  ARM A - baseline (current augmentation), seeds 0/1/2
echo  started %DATE% %TIME%
echo ==========================================================
call "%TRAIN%" medium --from augment --seed 0 --no-bench
if errorlevel 1 (echo [FAIL] A-s0 & set /a FAILS+=1)
call "%TRAIN%" medium --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] A-s1 & set /a FAILS+=1)
call "%TRAIN%" medium --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] A-s2 & set /a FAILS+=1)

echo.
echo ==========================================================
echo  ARM B - pad-then-mix, seeds 0/1/2
echo  %TIME%  (this rebuilds the data; arm A is already trained)
echo ==========================================================
call "%TRAIN%" medium --pad-then-mix --tag pad --from augment --seed 0 --no-bench
if errorlevel 1 (echo [FAIL] B-s0 & set /a FAILS+=1)
call "%TRAIN%" medium --pad-then-mix --tag pad --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] B-s1 & set /a FAILS+=1)
call "%TRAIN%" medium --pad-then-mix --tag pad --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] B-s2 & set /a FAILS+=1)

echo.
echo ==========================================================
echo  BENCH - real voice, real room, recall vs SNR
echo  %TIME%
echo ==========================================================
call "%BENCH%" --snr-sweep
if errorlevel 1 (echo [FAIL] bench & set /a FAILS+=1)

echo.
echo ==========================================================
echo  DONE %DATE% %TIME%   failed steps: %FAILS%
echo ==========================================================
echo  Compare the medium@s* rows against the medium@pad-s* rows as two
echo  GROUPS. The fix counts only if arm B's range clears arm A's range at
echo  +10 and +5 dB. Overlapping ranges mean no result, whatever the single
echo  best row says.
echo.
endlocal
