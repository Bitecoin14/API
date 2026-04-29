@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Hand Tracker - First-Time Setup
echo ============================================================
echo.

:: ── 1. Locate Python 3.9+ ────────────────────────────────────
:: Try explicit version launchers first (py -3.x), then fall back to python/python3.
:: This avoids triggering the Windows Store stub that opens the Store instead of running.

set PYTHON=

:: Try the Python Launcher (py.exe) which is the most reliable on Windows
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%V in ('py --version 2^>^&1') do (
        for /f "tokens=1,2 delims=." %%M in ("%%V") do (
            if %%M geq 3 if %%N geq 9 set PYTHON=py
        )
    )
)

:: Fall back to python.exe — skip Windows Store stubs (they return exit code 9009)
if not defined PYTHON (
    where python >nul 2>&1
    if not errorlevel 1 (
        python --version >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do (
                for /f "tokens=1,2 delims=." %%M in ("%%V") do (
                    if %%M geq 3 if %%N geq 9 set PYTHON=python
                )
            )
        )
    )
)

:: Last resort: python3
if not defined PYTHON (
    where python3 >nul 2>&1
    if not errorlevel 1 (
        python3 --version >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=2 delims= " %%V in ('python3 --version 2^>^&1') do (
                for /f "tokens=1,2 delims=." %%M in ("%%V") do (
                    if %%M geq 3 if %%N geq 9 set PYTHON=python3
                )
            )
        )
    )
)

if not defined PYTHON (
    echo [ERROR] Python 3.9 or higher was not found on this machine.
    echo.
    echo  Please install Python from https://www.python.org/downloads/
    echo  Make sure to tick "Add Python to PATH" during installation.
    echo  After installing, close this window and run setup.bat again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%V in ('!PYTHON! --version 2^>^&1') do set PY_VER=%%V
echo [OK] Found Python !PY_VER! (!PYTHON!)
echo.

:: ── 2. Create virtual environment ────────────────────────────
set VENV_DIR=%~dp0.venv

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists, skipping creation.
) else (
    echo [..] Creating virtual environment in .venv ...
    !PYTHON! -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo  Try running: !PYTHON! -m ensurepip --upgrade
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)
echo.

:: ── 3. Install / upgrade dependencies ────────────────────────
echo [..] Installing packages (this may take a few minutes on first run) ...
echo.
call "%VENV_DIR%\Scripts\activate.bat"

python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARN] pip upgrade failed — continuing anyway.
)

:: Remove plain opencv-python if present to avoid conflicts with opencv-contrib-python
python -m pip show opencv-python >nul 2>&1
if not errorlevel 1 (
    echo [..] Removing conflicting opencv-python package ...
    python -m pip uninstall opencv-python -y --quiet
)

pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo  Double-click run.bat to start the hand tracker.
echo ============================================================
echo.
pause
endlocal
