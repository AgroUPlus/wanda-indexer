@echo off
REM Pull the Wanda database off the phone, index it here, push it back.
REM
REM Credentials are NOT stored in this file. Copy .env.example to .env and put
REM your Navidrome details there.
setlocal enabledelayedexpansion
title Wanda Fingerprint Indexer
cd /d "%~dp0"

set "DB_PATH=%~dp0wanda_music.db"
set "PACKAGE=com.wander.android.debug"

set "PYEXE="
where python >nul 2>nul && set "PYEXE=python"
if not defined PYEXE (
    where py >nul 2>nul && set "PYEXE=py -3"
)
if not defined PYEXE (
    echo [ERROR] Python 3 was not found on PATH.
    pause
    exit /b 1
)

echo ======================================================================
echo            Wanda Desktop Fingerprinting / Feature Indexer
echo ======================================================================
echo.
echo   Database : %DB_PATH%
echo   Package  : %PACKAGE%
echo.
echo   [1] Full sync   : pull from phone, index, push back
echo   [2] Index only  : use the local wanda_music.db
echo   [3] Pull only   : copy the database off the phone
echo   [4] Push only   : copy the local database to the phone
echo   [5] Check       : preflight only, changes nothing
echo   [6] Repair      : rebuild a corrupt database (keeps a backup)
echo ----------------------------------------------------------------------
set /p OPTION="Choose (1-6) [1]: "
if "%OPTION%"=="" set "OPTION=1"
echo.

if "%OPTION%"=="5" goto DO_CHECK
if "%OPTION%"=="6" goto DO_REPAIR
if "%OPTION%"=="4" goto DO_PUSH
if "%OPTION%"=="2" goto DO_INDEX
if "%OPTION%"=="1" goto DO_PULL
if "%OPTION%"=="3" goto DO_PULL
echo [ERROR] Unrecognised choice "%OPTION%".
pause
exit /b 1

:DO_CHECK
%PYEXE% main.py --db "%DB_PATH%" --check
goto FINISHED

:DO_REPAIR
%PYEXE% main.py --db "%DB_PATH%" --repair
goto FINISHED

:DO_PULL
echo [STEP] Pulling the database from the phone...
%PYEXE% -c "from core.db_sync import pull_database; import sys; sys.exit(0 if pull_database(r'%DB_PATH%', '%PACKAGE%') else 1)"
if errorlevel 1 (
    echo [ERROR] Pull failed. Unlock the phone and allow USB debugging, then retry.
    pause
    exit /b 1
)
if "%OPTION%"=="3" goto FINISHED

:DO_INDEX
echo.
echo [STEP] Indexing...
%PYEXE% main.py --db "%DB_PATH%"
if errorlevel 1 (
    echo.
    echo [WARNING] The indexer exited early. Progress up to the last checkpoint was saved.
    set /p CONTINUE="Push what has been indexed so far to the phone? (y/N): "
    if /i not "!CONTINUE!"=="y" goto FINISHED
)
if "%OPTION%"=="2" goto FINISHED

:DO_PUSH
echo.
echo [STEP] Pushing the database back to the phone...
%PYEXE% -c "from core.db_sync import push_database; import sys; sys.exit(0 if push_database(r'%DB_PATH%', '%PACKAGE%') else 1)"
if errorlevel 1 (
    echo [ERROR] Push failed.
    pause
    exit /b 1
)

:FINISHED
echo.
echo ======================================================================
echo Done.
echo ======================================================================
pause
