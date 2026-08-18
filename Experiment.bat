@echo off
REM Five recipes x four seeds, all pad-then-mix. One command, walk away, ~13.5 h.
REM
REM     Experiment.bat
REM
REM WHAT THE LAST ONE SETTLED, so this one does not re-ask it. pad-then-mix
REM removes the silence shortcut - positives used to carry ~0.9 s of leading
REM digital silence that openWakeWord's stream never provides. Benched
REM 2026-08-17 on real voice through the real runtime, recall with the phrase
REM said OVER room audio, each model at its own threshold:
REM
REM                              +20dB  +10dB   +5dB   0dB
REM     pad-then-mix (3 models)  69-81% 45-54% 35-47% 20-29%
REM     current data (2 models)  28-51% 12-27%  7-17%  5-12%
REM     v1.0 incumbent             46%    21%    15%     9%
REM
REM No overlap anywhere, and the two same-recipe seeds differ by only 6-12
REM points, so the gap clears seed noise. Four generations had been stuck at
REM +10 dB = 16-22%. THE CONTROL ARM IS NOT REPEATED HERE - ten models across
REM five generations agree on it, and v1.0 stays in the bench as the live
REM reference.
REM
REM WHAT THIS ONE ASKS. Two factors on data that no longer has the shortcut,
REM plus one architecture probe:
REM
REM   steps            100k vs 200k. 400k reversed sign once the data got
REM                    honest (alfred.yaml has the numbers), so the useful
REM                    range moved down, not up.
REM   negative weight  3000 vs 12000. Not a guess: trainer.py doubles the
REM                    weight when validation FPPH misses target, twice and no
REM                    more, and every pad model hit that ceiling and stayed
REM                    10-17x over budget. The knob was pinned.
REM   dnn head         one cell. jarvis - the model that behaves better in
REM                    noise - uses a DNN head, and our one previous try was
REM                    on shortcut data, which tested nothing.
REM
REM WHY FOUR SEEDS. livekit pins no RNG, so every run is a draw, and seed
REM variance has been larger than most effects chased here (clean 64% vs 86%
REM on byte-identical features). A cell is a DISTRIBUTION; compare ranges,
REM never best-vs-best. Four is what the budget bought after cutting the
REM control arm and 400k.
REM
REM ORDERING. Every model shares ONE dataset, so seed 0 builds it (--from
REM augment) and seeds 1-3 reuse it (--from train). All five variants train
REM inside one invocation per seed, which is what makes them paired.
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
REM ~18 superseded artifacts at ~6 min each. The _v1.2 rows are last night's
REM survivors, kept because bench_real.py REWRITES its results file rather than
REM merging - anything unlisted vanishes from it. medium_pad_s1_v1.2 is also
REM the model most likely to be deployed when this starts, so it is the row
REM every new one has to beat; against it, medium_s2_v1.2 is the same recipe
REM WITHOUT pad-then-mix, and the pair is the effect this sweep builds on.
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
