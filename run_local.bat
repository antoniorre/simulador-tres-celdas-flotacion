@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  echo No se encontro Python. Instale Python 3.10 o superior desde https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno virtual local...
  py -m venv .venv
)

echo Instalando/verificando librerias...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Abriendo simulador en http://127.0.0.1:8504
call ".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8504
pause
