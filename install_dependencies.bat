@echo off
echo ===============================================
echo  Drone Optimization Project - Setup Installer
echo  One-click installation for all dependencies
echo  (Python venv + pip install requirements)
echo ===============================================
echo.

REM Check if Python exists
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found! Install Python 3.10+ first.
    pause
    exit /b
)

echo ✅ Python detected.

REM Create virtual environment
echo.
echo Creating Python virtual environment 'venv'...
python -m venv venv

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b
)

echo ✅ Virtual environment created.

REM Activate environment and install requirements
echo.
echo Activating environment and installing dependencies...
call venv\Scripts\activate.bat

IF EXIST "requirements.txt" (
    echo ✅ requirements.txt found. Installing...
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
) ELSE (
    echo ❌ requirements.txt NOT found!
    echo Creating default one...
    (
        echo numpy
        echo scipy
        echo pandas
        echo scikit-learn
        echo matplotlib
        echo joblib
        echo tk
    ) > requirements.txt
    pip install -r requirements.txt
)

echo.
echo ✅ All dependencies installed successfully!
echo -----------------------------------------------
echo To start your project, run:
echo     venv\Scripts\activate
echo     python main.py
echo -----------------------------------------------
pause
