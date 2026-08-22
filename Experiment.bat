@echo off
REM Five recipes x four seeds, all pad-then-mix. One command, walk away, ~13.5 h.
REM
REM     Experiment.bat
REM
REM pad-then-mix removes the silence shortcut - positives used to carry ~0.9 s
REM of leading digital silence that openWakeWord's stream never provides.
REM Benched 2026-08-17 on real voice through the real runtime, recall with the
REM phrase said OVER room audio, each model at its own threshold:
REM
REM                              +20dB  +10dB   +5dB   0dB
REM     pad-then-mix (3 models)  69-81% 45-54% 35-47% 20-29%
REM     current data (2 models)  28-51% 12-27%  7-17%  5-12%
REM     v1.0 incumbent             46%    21%    15%     9%
REM
REM No overlap, and same-recipe seeds differ by only 6-12 points. Four
REM generations had been stuck at +10 dB = 16-22%. The control arm is NOT
REM repeated here; v1.0 stays in the bench as the live reference.
REM
REM Factors, on data that no longer has the shortcut:
REM
REM   steps            100k vs 200k. 400k reversed sign once the data got
REM                    honest (alfred.yaml has the numbers).
REM   negative weight  3000 vs 12000. trainer.py doubles the weight when
REM                    validation FPPH misses target, twice and no more, and
REM                    every pad model hit that ceiling 10-17x over budget.
REM   dnn head         one cell. jarvis uses a DNN head; our one previous try
REM                    was on shortcut data.
REM
REM FOUR SEEDS: livekit pins no RNG and seed variance has beaten most effects
REM chased here (clean 64% vs 86% on byte-identical features) - a cell is a
REM DISTRIBUTION, so compare ranges, never best-vs-best.
REM
REM ORDERING: all models share ONE dataset, so seed 0 builds it (--from augment)
REM and seeds 1-3 reuse it (--from train). All five variants train in one
REM invocation per seed, which is what pairs them.
setlocal
set TRAIN=%~dp0Train.bat
set BENCH=%~dp0Bench.bat
set FAILS=0
set A=C:\Users\tillm\wake\artifacts
set V=C:\Users\tillm\projects\slopstation\k15\voice\models
set CELLS=medium medium-200k medium-nw medium-200k-nw dnn-medium

echo.
echo ============================================================
echo  PAD-THEN-MIX SWEEP : %CELLS%
echo  seeds 0,1,2,3   started %DATE% %TIME%
echo ============================================================
call "%TRAIN%" %CELLS% --pad-then-mix --tag pad --from augment --seed 0 --no-bench
if errorlevel 1 (echo [FAIL] s0 & set /a FAILS+=1)
call "%TRAIN%" %CELLS% --pad-then-mix --tag pad --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] s1 & set /a FAILS+=1)
call "%TRAIN%" %CELLS% --pad-then-mix --tag pad --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] s2 & set /a FAILS+=1)
call "%TRAIN%" %CELLS% --pad-then-mix --tag pad --from train --seed 3 --no-bench
if errorlevel 1 (echo [FAIL] s3 & set /a FAILS+=1)

echo.
echo ============================================================
echo  BENCH - real voice, real room, recall vs SNR
echo  %TIME%
echo ============================================================
REM The 20 new models, listed explicitly: the default glob would also re-score
REM ~18 superseded artifacts at ~6 min each. bench_real.py REWRITES its results
REM file rather than merging, so anything unlisted vanishes from it - hence the
REM _v1.2 rows: medium_pad_s1_v1.2 is the row every new one has to beat, and
REM medium_s2_v1.2 is the same recipe WITHOUT pad-then-mix.
call "%BENCH%" --snr-sweep --models ^
 "%A%\hey_alfred_medium_pad_s0_v1.3.onnx"        "%A%\hey_alfred_medium_pad_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium_pad_s2_v1.3.onnx"        "%A%\hey_alfred_medium_pad_s3_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k_pad_s0_v1.3.onnx"   "%A%\hey_alfred_medium-200k_pad_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k_pad_s2_v1.3.onnx"   "%A%\hey_alfred_medium-200k_pad_s3_v1.3.onnx" ^
 "%A%\hey_alfred_medium-nw_pad_s0_v1.3.onnx"     "%A%\hey_alfred_medium-nw_pad_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium-nw_pad_s2_v1.3.onnx"     "%A%\hey_alfred_medium-nw_pad_s3_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k-nw_pad_s0_v1.3.onnx" "%A%\hey_alfred_medium-200k-nw_pad_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k-nw_pad_s2_v1.3.onnx" "%A%\hey_alfred_medium-200k-nw_pad_s3_v1.3.onnx" ^
 "%A%\hey_alfred_dnn-medium_pad_s0_v1.3.onnx"    "%A%\hey_alfred_dnn-medium_pad_s1_v1.3.onnx" ^
 "%A%\hey_alfred_dnn-medium_pad_s2_v1.3.onnx"    "%A%\hey_alfred_dnn-medium_pad_s3_v1.3.onnx" ^
 "%A%\hey_alfred_medium_pad_s1_v1.2.onnx"        "%A%\hey_alfred_medium_s2_v1.2.onnx" ^
 "%V%\hey_alfred_v1.0.onnx"
if errorlevel 1 (echo [FAIL] bench & set /a FAILS+=1)

echo.
echo ============================================================
echo  DONE %DATE% %TIME%   failed steps: %FAILS%
echo ============================================================
echo  Five cells of four seeds. Read them as RANGES:
echo     medium_pad_s*           100k, negative weight 3000
echo     medium-200k_pad_s*      200k, negative weight 3000
echo     medium-nw_pad_s*        100k, negative weight 12000
echo     medium-200k-nw_pad_s*   200k, negative weight 12000
echo     dnn-medium_pad_s*       100k, DNN head
echo.
echo  The row to beat is hey_alfred_medium_pad_s1_v1.2 - same idea, trained
echo  before the clean quarter stopped being digital silence. A cell only wins
echo  if its RANGE clears it; overlapping ranges mean no result, whatever the
echo  single best row says.
echo.
echo  Then re-run the top two or three with --target-fa 0.5 (about 6 min each).
echo  One hour of negatives cannot resolve a rate below 1/hr, so the table's
echo  default budget accepts a model with a false accept in it; --target-fa 0.5
echo  forces ZERO events and gives the number you would actually ship.
echo.
endlocal
