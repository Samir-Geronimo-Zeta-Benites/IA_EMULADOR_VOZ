@echo off
title VoiceMod GPU Setup
cd /d "%~dp0.."

echo =========================================
echo   VoiceMod - GPU Setup
echo =========================================
echo.

python --version >nul 2>&1 || (echo [X] Necesitas Python 3.10+ && pause && exit /b 1)

echo [1/3] Creando entorno virtual...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/3] Instalando dependencias GPU...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements\base.txt

echo [3/3] Descargando modelo base RVC...
python scripts\download_models.py

echo.
echo =========================================
echo   LISTO. Ahora pone tu audio en
echo   voces_audios_crudos/ y ejecuta:
echo.
echo   python scripts\train_gpu.py
echo.
echo   Luego python run.py
echo =========================================
pause
