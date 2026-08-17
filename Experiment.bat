@echo off
REM 2x2 factorial: {current data, pad-then-mix} x {100k steps, LONG steps},
REM three seeds per cell. One command, walk away, ~12.5 h.
REM
REM     Experiment.bat
REM
REM WHY A FACTORIAL RATHER THAN TWO SEPARATE A/Bs. pad-then-mix REMOVES a
REM shortcut - positives currently carry 912 ms of leading digital silence that
REM the deployed stream never provides, so the model can partly separate classes
REM without hearing the phrase. Take that away and it has to learn the harder
REM real task, which plausibly needs more steps to converge. Testing the fix at
REM 100k only could therefore show nothing while the fix is actually working.
REM The same interaction already appeared once: the -10 dB dataset was terrible
REM at 100k (snr-neg10, 14% @+10dB) and best-in-class at 400k (medium-400k,
REM 29%). Steps and data difficulty are not independent here.
REM
REM WHY THREE SEEDS. livekit pins no RNG, so every run is a random draw. Two
REM models on BYTE-IDENTICAL features with the same recipe scored clean 64% vs
REM 86% and +10 dB recall 14% vs 37% (2026-08-17). Seed variance is larger than
REM every effect this project has chased, which is why single-run conclusions
REM have had to be retracted twice. A cell is a DISTRIBUTION; one model is an
REM anecdote. Compare ranges, never best-vs-best.
REM
REM TO HALVE THE RUNTIME set LONG=medium-200k below (~8.3 h). 400k is the
REM default because 200k->400k is the step increase that has NOT been tested on
REM honest data, and the extremes maximise the chance of seeing a trend at all.
set LONG=medium-400k

REM ORDERING IS LOAD-BEARING. Both arms share one output directory, so arm A
REM must finish TRAINING before arm B's augment overwrites the features. Each
REM arm rebuilds once (--from augment) and its later seeds reuse that data
REM (--from train). Both variants train inside ONE invocation per seed, so they
REM are guaranteed byte-identical data AND a paired seed. Do not reorder.
setlocal
set TRAIN=%~dp0Train.bat
set BENCH=%~dp0Bench.bat
set FAILS=0
set A=C:\Users\tillm\wake\artifacts
set V=C:\Users\tillm\projects\slopstation\k15\voice\models

echo.
echo ============================================================
echo  ARM A - current augmentation : medium + %LONG% : seeds 0,1,2
echo  started %DATE% %TIME%
echo ============================================================
call "%TRAIN%" medium %LONG% --from augment --seed 0 --no-bench
if errorlevel 1 (echo [FAIL] A-s0 & set /a FAILS+=1)
call "%TRAIN%" medium %LONG% --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] A-s1 & set /a FAILS+=1)
call "%TRAIN%" medium %LONG% --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] A-s2 & set /a FAILS+=1)

echo.
echo ============================================================
echo  ARM B - pad-then-mix : medium + %LONG% : seeds 0,1,2
echo  %TIME%  (rebuilds the data; arm A is already trained)
echo ============================================================
call "%TRAIN%" medium %LONG% --pad-then-mix --tag pad --from augment --seed 0 --no-bench
if errorlevel 1 (echo [FAIL] B-s0 & set /a FAILS+=1)
call "%TRAIN%" medium %LONG% --pad-then-mix --tag pad --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] B-s1 & set /a FAILS+=1)
call "%TRAIN%" medium %LONG% --pad-then-mix --tag pad --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] B-s2 & set /a FAILS+=1)

echo.
echo ============================================================
echo  BENCH - real voice, real room, recall vs SNR
echo  %TIME%
echo ============================================================
REM The 12 new models plus the deployed incumbent, listed explicitly. The
REM default glob would also re-score ~14 superseded artifacts at ~6 min each,
REM adding well over an hour for rows already in bench_results.json.
call "%BENCH%" --snr-sweep --models ^
 "%A%\hey_alfred_medium_s0_v1.2.onnx" "%A%\hey_alfred_medium_s1_v1.2.onnx" "%A%\hey_alfred_medium_s2_v1.2.onnx" ^
 "%A%\hey_alfred_%LONG%_s0_v1.2.onnx" "%A%\hey_alfred_%LONG%_s1_v1.2.onnx" "%A%\hey_alfred_%LONG%_s2_v1.2.onnx" ^
 "%A%\hey_alfred_medium_pad_s0_v1.2.onnx" "%A%\hey_alfred_medium_pad_s1_v1.2.onnx" "%A%\hey_alfred_medium_pad_s2_v1.2.onnx" ^
 "%A%\hey_alfred_%LONG%_pad_s0_v1.2.onnx" "%A%\hey_alfred_%LONG%_pad_s1_v1.2.onnx" "%A%\hey_alfred_%LONG%_pad_s2_v1.2.onnx" ^
 "%V%\hey_alfred_v1.0.onnx"
if errorlevel 1 (echo [FAIL] bench & set /a FAILS+=1)

echo.
echo ============================================================
echo  DONE %DATE% %TIME%   failed steps: %FAILS%
echo ============================================================
echo  Four cells of three seeds. Read them as RANGES:
echo     medium_s*          current data, 100k
echo     %LONG%_s*     current data, long
echo     medium_pad_s*      pad-then-mix, 100k
echo     %LONG%_pad_s* pad-then-mix, long
echo.
echo  pad-then-mix wins only if its range clears the matching baseline range
echo  at +10 and +5 dB. Overlapping ranges mean NO result, whatever the single
echo  best row says. If the fix helps only in the long cells, that is the
echo  steps-by-difficulty interaction this design exists to catch.
echo.
endlocal
