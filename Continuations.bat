@echo off
REM Run-on positives: does teaching the phrase MID-SENTENCE fix the delivery
REM style that currently scores 0.011-0.041? Two cells x four seeds, ~8.5 h.
REM
REM     Continuations.bat
REM
REM WHAT IS BROKEN. Every TTS positive this repo has ever trained on is the
REM bare phrase followed by nothing. The score crests about a second AFTER the
REM talker stops - "alfred" ends on a low-energy /d/ - so a command said in one
REM breath stomps the exact window the model needs. Measured on v1.0, all
REM user-confirmed true positives: "hey alfred play hades" scores 0.011-0.041
REM against a 0.678 median for isolated utterances. Same class of defect as the
REM silence shortcut: the training window is a shape the microphone never sees.
REM
REM WHAT CHANGES. alfred.yaml's continuation_phrases (12 forms, first words
REM taken from grammar.yaml's real command vocabulary) join the positives at a
REM 50/50 split with the bare forms - pipeline.py weights them by repetition
REM because piper's sampler is a flat cycle over the phrase list.
REM
REM THE TRAP THAT WAS MEASURED AND DISARMED FIRST. livekit builds adversarial
REM negatives by substituting one word of each target phrase. Hand it "hey
REM alfred play" and it emits "hey alfred clay", "hey alfred nintendo", "hey
REM alfred amigo" - 39,098 of 133,713 phrases, 29% of the negative corpus,
REM every one containing the COMPLETE wake phrase and labelled NOT a wake word.
REM patch_adversarial_from_bare keeps the substitution generator on the two
REM bare forms; drilled after the patch, 48 of 17,124, the same benign
REM "hey al frederick" residue today's config already has.
REM
REM NO VERSION BUMP, deliberately. The data recipe changed, which normally
REM bumps VERSION - but --tag cont already makes every filename and every
REM results key distinct, and bumping mid-flight would have stranded the v1.3
REM sweep's bench list. The tag exists for exactly this.
setlocal
set TRAIN=%~dp0Train.bat
set BENCH=%~dp0Bench.bat
set FAILS=0
set A=C:\Users\tillm\wake\artifacts
set V=C:\Users\tillm\projects\slopstation\k15\voice\models
set RUNONS=C:\Users\tillm\wake\bench\runons

REM ---- SET THESE TWO FROM THE v1.3 BENCH BEFORE LAUNCHING ----------------
REM CELLS: the recipes that won on real audio. Default is the two negative-
REM weight-3000 cells, which the synthetic eval favoured at 60%% through the
REM sweep - confirm against the bench table, since it has reversed that eval
REM twice. REF: the best v1.3 model, the row every continuation model must
REM beat. It is the SAME recipe with bare-phrase positives, so the pair is the
REM experiment.
set CELLS=medium medium-200k
set REF=%A%\hey_alfred_medium_pad_s0_v1.3.onnx
REM -----------------------------------------------------------------------

echo.
echo ============================================================
echo  CONTINUATION POSITIVES : %CELLS% : seeds 0,1,2,3
echo  started %DATE% %TIME%
echo ============================================================
REM --from generate, once: the phrase list changed, so pipeline.py clears the
REM TTS splits and resynthesises (~40-70 min) rather than counting the old
REM clips as "already complete" and silently training on bare-phrase data.
REM Seeds 1-3 reuse that corpus, so all four are the same data.
call "%TRAIN%" %CELLS% --continuations --pad-then-mix --tag cont --from generate --seed 0 --no-bench
if errorlevel 1 (echo [FAIL] s0 & set /a FAILS+=1)
call "%TRAIN%" %CELLS% --continuations --pad-then-mix --tag cont --from train --seed 1 --no-bench
if errorlevel 1 (echo [FAIL] s1 & set /a FAILS+=1)
call "%TRAIN%" %CELLS% --continuations --pad-then-mix --tag cont --from train --seed 2 --no-bench
if errorlevel 1 (echo [FAIL] s2 & set /a FAILS+=1)
call "%TRAIN%" %CELLS% --continuations --pad-then-mix --tag cont --from train --seed 3 --no-bench
if errorlevel 1 (echo [FAIL] s3 & set /a FAILS+=1)

echo.
echo ============================================================
echo  BENCH 1 of 2 - the standard 50 isolated utterances
echo  %TIME%   this is the REGRESSION check, not the result
echo ============================================================
REM Continuations must not cost anything on the delivery that already works.
call "%BENCH%" --snr-sweep --models ^
 "%A%\hey_alfred_medium_cont_s0_v1.3.onnx"      "%A%\hey_alfred_medium_cont_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium_cont_s2_v1.3.onnx"      "%A%\hey_alfred_medium_cont_s3_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k_cont_s0_v1.3.onnx" "%A%\hey_alfred_medium-200k_cont_s1_v1.3.onnx" ^
 "%A%\hey_alfred_medium-200k_cont_s2_v1.3.onnx" "%A%\hey_alfred_medium-200k_cont_s3_v1.3.onnx" ^
 "%REF%" "%V%\hey_alfred_v1.0.onnx"
if errorlevel 1 (echo [FAIL] bench-isolated & set /a FAILS+=1)

echo.
echo ============================================================
echo  BENCH 2 of 2 - the run-on utterances : THE RESULT
echo  %TIME%
echo ============================================================
REM Without this the experiment is unfalsifiable: only about two of the 50
REM standard positives are run-ons, and two clips cannot move a number whose
REM 95%% interval is +/-13 points. Record ~25 of them first - the header of
REM k15\voice\bench\slice_utterances.py has the drill.
if not exist "%RUNONS%" (
  echo.
  echo   *** NO RUN-ON SET at %RUNONS% - SKIPPING THE ONLY BENCH THAT
  echo       MEASURES WHAT THIS RUN CHANGED. The isolated table above is a
  echo       regression check and nothing more. Record ~25 run-ons, slice
  echo       them into that folder, and re-run this bench by hand:
  echo.
  echo       Bench.bat --snr-sweep --positives %RUNONS% --models ...
  echo.
) else (
  call "%BENCH%" --snr-sweep --positives "%RUNONS%" --json C:\Users\tillm\wake\bench_runons.json --models ^
   "%A%\hey_alfred_medium_cont_s0_v1.3.onnx"      "%A%\hey_alfred_medium_cont_s1_v1.3.onnx" ^
   "%A%\hey_alfred_medium_cont_s2_v1.3.onnx"      "%A%\hey_alfred_medium_cont_s3_v1.3.onnx" ^
   "%A%\hey_alfred_medium-200k_cont_s0_v1.3.onnx" "%A%\hey_alfred_medium-200k_cont_s1_v1.3.onnx" ^
   "%A%\hey_alfred_medium-200k_cont_s2_v1.3.onnx" "%A%\hey_alfred_medium-200k_cont_s3_v1.3.onnx" ^
   "%REF%" "%V%\hey_alfred_v1.0.onnx"
  if errorlevel 1 (echo [FAIL] bench-runons & set /a FAILS+=1)
)

echo.
echo ============================================================
echo  DONE %DATE% %TIME%   failed steps: %FAILS%
echo ============================================================
echo  Two tables, and they answer different questions:
echo.
echo    RUN-ONS   does the continuation cell's RANGE clear %REF%'s?
echo              That is the whole point of the run.
echo    ISOLATED  did it cost anything on the delivery that already worked?
echo              Overlapping ranges here is the GOOD outcome.
echo.
echo  A win on run-ons with a loss on isolated is not a win - the couch says
echo  the phrase both ways.
echo.
endlocal
