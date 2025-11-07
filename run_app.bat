@echo off
echo ===============================================
echo     Drone Optimization Project - RUNNER
echo ===============================================
echo.

REM Check for virtual environment
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run install_dependencies.bat first.
    pause
    exit /b
)

echo ✅ Virtual environment found.

REM Activate environment
echo Activating environment...
call venv\Scripts\activate.bat

REM Check for main.py
IF NOT EXIST "main.py" (
    echo [ERROR] main.py not found in the project root.
    pause
    exit /b
)

echo ✅ Starting the application...
echo -----------------------------------------------
python main.py
echo -----------------------------------------------
echo ✅ Application closed. Virtual environment remains active.
echo Close this window or type 'deactivate' to exit venv.
pause
