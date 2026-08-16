@echo off
REM Rank trained candidates on REAL audio - your voice, your room, through
REM openWakeWord. THIS is the eval that picks a model; the table Train.bat
REM prints is saturated and ranks nothing.
REM
REM     Bench.bat                          every .onnx in <root>\artifacts
REM     Bench.bat --target-fa 0.5          tighter false-accept budget
REM     Bench.bat --models a.onnx b.onnx   just these two
REM
REM Needs <root>\bench\positives (one utterance per wav, from
REM k15\voice\bench\slice_utterances.py) and <root>\data\heldout (room and
REM game audio with NO wake word in it).
setlocal
if "%WAKE_VENV%"=="" set WAKE_VENV=C:\Users\tillm\wake\.venv
if not exist "%WAKE_VENV%\Scripts\python.exe" (
  echo [Bench] no venv at %WAKE_VENV%
  exit /b 1
)
"%WAKE_VENV%\Scripts\python.exe" "%~dp0bench_real.py" %*
