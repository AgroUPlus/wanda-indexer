@echo off
REM Ships the indexed database to the debug build.
REM
REM The app is stopped before the write and the stale -wal/-shm are removed one per command:
REM a running app keeps writing its own write-ahead log, and a log left beside a replaced
REM database is replayed onto pages that no longer exist. SQLite then reports corruption and
REM Room's DefaultDatabaseErrorHandler deletes the database -- a push that looks successful and
REM destroys the library. Grouped `rm a b` fails on this device shell with "Needs 1 argument".
REM Run from wherever this script lives, so the checkout can move.
cd /d "%~dp0"
set ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe
set PKG=com.wander.android.debug

"%ADB%" shell am force-stop %PKG%
"%ADB%" shell am kill %PKG%
ping -n 3 127.0.0.1 >nul

"%ADB%" exec-in run-as %PKG% sh -c "cat > databases/wanda_music.db" < wanda_push.db
if errorlevel 1 goto failed

"%ADB%" shell run-as %PKG% rm -f databases/wanda_music.db-wal
"%ADB%" shell run-as %PKG% rm -f databases/wanda_music.db-shm

echo == on device ==
"%ADB%" shell run-as %PKG% ls -l databases/wanda_music.db
echo Reopen Wanda on the phone, then re-run this script's last line to confirm the size held.
goto :eof

:failed
echo PUSH FAILED
