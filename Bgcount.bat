@echo off
REM Run 2, PRE-STAGED and GATED: does drawing more background negatives per
REM build recover the variety the pool just gained? One cell x four seeds,
REM ~8-9 h. DO NOT LAUNCH until both gates below pass.
REM
REM     Bgcount.bat
REM
REM THE HYPOTHESIS. n_background_samples: 2000 was sized when the background
REM pool held 2,274 files - a build drew 88%% of the pool's variety. The
REM 2026-08-20 recording campaign grew the pool to 6,324 files (sess5, sess6,
REM room-dup, mined hard\) and the same 2,000-draw now samples 32%%. Doubling
REM to 4,000 (val 400 -> 800, scaled with it) puts a build back at 63%%.
REM This is the ONE knob with a genuinely new justification after the data
REM change; every other axis was settled by the 20-model sweep.
REM
REM LAUNCH GATE, decided by the pad2 bench, not by this file: run 2 exists
REM only if NO pad2 seed holds zero events over the 6-h heldout at a
REM threshold keeping +10 dB recall >= ~50%%. If one does, deploy it and
REM delete this file - that is its delete condition.
REM
REM WHY THIS EDITS THE BASE YAML INSTEAD OF ADDING A VARIANT. pipeline.py
REM rejects variant overrides outside TRAINING_ONLY, deliberately: variants
REM share the data stages, so a variant touching a data key would silently
REM train on the previous variant's data. Its own comment says the sweep
REM mechanism for data keys is "edit alfred.yaml and run the whole pipeline
REM again" - this file is that, with the edit checked rather than trusted:
REM
REM     alfred.yaml:  n_background_samples: 2000      ->  4000
REM                   n_background_samples_val: 400   ->  800
REM
REM THE STAMP TRAP, third sighting. The stale-data stamp records SNR, clean
REM fraction, RIR, rounds, pad-then-mix and phrases - NOT n_background_samples.
REM After the yaml edit, --from train would report "SAME as this run" and
REM silently reuse the 2,000-draw features. Seed 0 therefore MUST run
REM --from augment, and a seed-0 failure MUST abort the chain: seeds 1-3
REM falling back to run-1's features would train mislabeled models, wrong
REM quietly, not loudly.
REM
REM HELD STILL, deliberately: max_negative_weight 12000 (it interacts with
REM steps - the sweep measured 12000 worst at 100k and best at 200k, so no
REM second knob moves in the same run) and steps 200000. Tag bg4k keeps every
REM stem and results key distinct from the pad2 set; no VERSION bump, per the
REM Continuations.bat precedent - the tag is the experiment id.
setlocal
set TRAIN=%~dp0Train.bat
set BENCH=%~dp0Bench.bat
set YAML=%~dp0alfred.yaml
set A=C:\Users\tillm\wake\artifacts
set V=C:\Users\tillm\projects\slopstation\k15\voice\models
set FAILS=0

REM ---- GATE 1: run 1 actually finished (its last seed's artifact exists) --
if not exist "%A%\hey_alfred_medium-200k-nw_pad2_s3_v1.3.onnx" (
  echo [Bgcount] GATE 1 FAILED: pad2 s3 artifact missing - run 1 is still
  echo           training or died. This run's whole point is comparing
  echo           against the finished pad2 distribution. Aborting.
  exit /b 1
)
REM ---- GATE 2: the yaml edit is actually in place -------------------------
findstr /C:"n_background_samples: 4000" "%YAML%" >nul || (
  echo [Bgcount] GATE 2 FAILED: alfred.yaml still has the old draw count.
  echo           Edit it first:  n_background_samples: 2000  -^>  4000
  echo                           n_background_samples_val: 400  -^>  800
  exit /b 1
)
findstr /C:"n_background_samples_val: 800" "%YAML%" >nul || (
  echo [Bgcount] GATE 2 FAILED: n_background_samples_val not scaled to 800.
  exit /b 1
)

echo.
echo ============================================================
echo  BG-COUNT 2000 -^> 4000 : medium-200k-nw : seeds 0,1,2,3
echo  started %DATE% %TIME%
echo ============================================================
call "%TRAIN%" medium-200k-nw --pad-then-mix --tag bg4k --from augment --seed 0 --no-bench
if errorlevel 1 (
  echo [Bgcount] seed 0 FAILED during the data rebuild. Seeds 1-3 would
  echo           silently reuse run-1 features - aborting instead.
  exit /b 1
)
call "%TRAIN%" medium-200k-nw --pad-then-mix --tag bg4k --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] s1 & set /a FAILS+=1)
call "%TRAIN%" medium-200k-nw --pad-then-mix --tag bg4k --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] s2 & set /a FAILS+=1)
call "%TRAIN%" medium-200k-nw --pad-then-mix --tag bg4k --from train --seed 3 --no-bench
if errorlevel 1 (echo [FAIL] s3 & set /a FAILS+=1)

echo.
echo ============================================================
echo  BENCH : bg4k vs pad2 vs deployed vs v1.0, one honest table
echo  %TIME%
echo ============================================================
call "%BENCH%" --snr-sweep --json C:\Users\tillm\wake\bench_bg4k.json --models ^
 "%A%\hey_alfred_medium-200k-nw_bg4k_s0_v1.3.onnx" "%A%\hey_alfred_medium-200k-nw_bg4k_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k-nw_bg4k_s2_v1.3.onnx" "%A%\hey_alfred_medium-200k-nw_bg4k_s3_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k-nw_pad2_s0_v1.3.onnx" "%A%\hey_alfred_medium-200k-nw_pad2_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k-nw_pad2_s2_v1.3.onnx" "%A%\hey_alfred_medium-200k-nw_pad2_s3_v1.3.onnx" ^
 "%V%\hey_alfred_v1.3-200knw.onnx" "%V%\hey_alfred_v1.0.onnx"
if errorlevel 1 (echo [FAIL] bench & set /a FAILS+=1)

echo.
echo ============================================================
echo  DONE %DATE% %TIME%   failed steps: %FAILS%
echo ============================================================
echo  Read it as DISTRIBUTION vs DISTRIBUTION - four bg4k seeds against four
echo  pad2 seeds. One lucky seed is the lesson of this project, not a result.
echo  Deploy gate unchanged: zero events over the 6-h heldout at a threshold
echo  holding +10 dB recall, with margin below the recall knee.
echo.
endlocal
