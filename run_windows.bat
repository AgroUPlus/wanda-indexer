@echo off
REM Wanda indexer launcher for Windows. Arguments pass through to main.py.
REM Paths are derived from this script's own location, so the folder can be
REM moved or used by a different user without editing anything.
setlocal
cd /d "%~dp0"

set "PYEXE="
where python >nul 2>nul && set "PYEXE=python"
if not defined PYEXE (
    where py >nul 2>nul && set "PYEXE=py -3"
)

if defined PYEXE (
    echo [EXEC] Using Windows Python: %PYEXE%
    %PYEXE% main.py %*
) else (
    where wsl >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Neither Windows Python nor WSL was found.
        echo         Install Python 3 from https://python.org and re-run.
        pause
        exit /b 1
    )
    echo [EXEC] Windows Python not found; running through WSL...
    wsl bash -lc "cd \"$(wslpath '%~dp0')\" && ./run.sh %*"
)

set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo [WARNING] Indexer exited with code %RC%.
pause
exit /b %RC%
