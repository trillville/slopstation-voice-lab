@echo off
REM Wake-model training, gaming PC only. Arguments pass straight through to
REM pipeline.py:
REM
REM     Train.bat                     every size, from scratch (hours)
REM     Train.bat medium              just medium
REM     Train.bat medium --from train reuse the clips already generated
REM     Train.bat --list              what is on disk, run nothing
REM
REM The venv lives with the DATA (torch+CUDA, 16 GB), not in this repo.
REM Override with set WAKE_VENV=... if it moves.
setlocal
if "%WAKE_VENV%"=="" set WAKE_VENV=C:\Users\tillm\wake\.venv
if not exist "%WAKE_VENV%\Scripts\python.exe" (
  echo [Train] no venv at %WAKE_VENV%
  echo [Train] set WAKE_VENV to the folder holding Scripts\python.exe
  exit /b 1
)
"%WAKE_VENV%\Scripts\python.exe" "%~dp0pipeline.py" %*
