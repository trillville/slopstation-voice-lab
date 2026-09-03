@echo off
REM Rank trained candidates on REAL audio - your voice, your room, through
REM openWakeWord. The table Train.bat prints is saturated and ranks nothing.
REM
REM     Bench.bat                          <root>\artifacts\*.onnx PLUS the
REM                                        vendored src\slopstation\agent\models\*.onnx -
REM                                        the incumbent is always in the race
REM     Bench.bat --target-fa 0.5          tighter false-accept budget
REM     Bench.bat --models a.onnx b.onnx   just these two
REM     Bench.bat --noise-only             negatives only, ranked by ceiling -
REM                                        the one comparison valid ACROSS
REM                                        phrases (jarvis vs alfred)
REM
REM Needs <root>\data\heldout (room/game audio, wake-phrase-free) and, unless
REM --noise-only, <root>\bench\positives (one utterance per wav, from
REM slopstation.agent.bench.slice_utterances).
setlocal
if "%WAKE_VENV%"=="" set WAKE_VENV=C:\Users\tillm\wake\.venv
if not exist "%WAKE_VENV%\Scripts\python.exe" (
  echo [Bench] no venv at %WAKE_VENV%
  exit /b 1
)
"%WAKE_VENV%\Scripts\python.exe" "%~dp0bench_real.py" %*
