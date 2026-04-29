@echo off
setlocal

set VENV_DIR=%~dp0.venv
set SCRIPT=%~dp0hand_tracker.py

:: ── Check setup has been run ──────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo  Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

:: ── Activate venv and launch ──────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

:: Verify the python inside the venv is actually usable
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Virtual environment python is not working.
    echo  Please delete the .venv folder and run setup.bat again.
    echo.
    pause
    exit /b 1
)

python "%SCRIPT%" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Hand tracker exited with an error. See output above.
    pause
)

endlocal
