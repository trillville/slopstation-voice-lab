@echo off
REM Wake-model training, gaming PC only. Everything after Train.bat is passed
REM straight through to pipeline.py, so:
REM
REM     Train.bat                     every size, from scratch (hours)
REM     Train.bat medium              just medium
REM     Train.bat medium --from train reuse the clips already generated
REM     Train.bat --list              what is on disk, run nothing
REM
REM The venv lives with the DATA, not with this repo: it carries torch+CUDA and
REM the data folder is 16 GB, so neither belongs in a checkout that also gets
REM pulled onto the K15. Override with set WAKE_VENV=... if it ever moves.
setlocal
if "%WAKE_VENV%"=="" set WAKE_VENV=C:\Users\tillm\wake\.venv
if not exist "%WAKE_VENV%\Scripts\python.exe" (
  echo [Train] no venv at %WAKE_VENV%
  echo [Train] set WAKE_VENV to the folder holding Scripts\python.exe
  exit /b 1
)
"%WAKE_VENV%\Scripts\python.exe" "%~dp0pipeline.py" %*
