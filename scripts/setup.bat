@echo off
title VoiceMod - Instalacion
setlocal enabledelayedexpansion

cd /d "%~dp0.."

echo =========================================
echo   VoiceMod - Instalacion Automatica
echo =========================================
echo.

:: Ensure Python is in PATH (fix Microsoft Store alias issue)
set "PYTHON_DIR=C:\Users\Developer\AppData\Local\Programs\Python\Python311"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python no encontrado.
    echo     Buscalo en: %PYTHON_DIR%
    pause
    exit /b 1
)

echo [OK] Python detectado:
python --version
echo.

:: Create virtual environment
echo [1/4] Creando entorno virtual...
if exist .venv (
    echo       Entorno ya existe, limpiando...
    rmdir /s /q .venv
)
python -m venv .venv
call .venv\Scripts\activate.bat
echo.

:: Upgrade pip
echo [2/4] Actualizando pip...
python -m pip install --upgrade pip -q
echo.

:: Install dependencies
echo [3/4] Instalando dependencias (esto toma ~5-10 min)...
pip install -r requirements\base.txt
if errorlevel 1 (
    echo [X] Error instalando dependencias
    pause
    exit /b 1
)
echo.
echo   Dependencias instaladas correctamente.

:: Download base models
echo [4/4] Verificando modelos base...
python scripts\download_models.py
echo.

echo =========================================
echo   Instalacion completada!
echo =========================================
echo.
echo Para iniciar el programa:
echo   run.bat
echo.

pause
