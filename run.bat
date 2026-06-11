@echo off
cd /d "%~dp0"
set "PYTHON_DIR=C:\Users\Developer\AppData\Local\Programs\Python\Python311"
set "FFMPEG_DIR=C:\Users\Developer\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%FFMPEG_DIR%;%PATH%"
call .venv\Scripts\activate.bat 2>nul
python run.py %*
if errorlevel 1 pause
