@echo off
REM Build the validation set from HELD-OUT room + game audio, so the FPPH the
REM trainer reports is a number about your living room rather than about
REM livekit's synthetic negatives. Run this BEFORE Train.bat.
REM
REM     Validate.bat              featurize <root>\data\heldout\**.wav
REM     Validate.bat --restore    put livekit's stock set back
setlocal
if "%WAKE_VENV%"=="" set WAKE_VENV=C:\Users\tillm\wake\.venv
if not exist "%WAKE_VENV%\Scripts\python.exe" (
  echo [Validate] no venv at %WAKE_VENV%
  exit /b 1
)
"%WAKE_VENV%\Scripts\python.exe" "%~dp0make_validation.py" %*
