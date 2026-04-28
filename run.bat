@echo off
chcp 65001 >nul 2>&1
:: ETF Analyzer -- Launcher
:: Usa el Python de Anaconda directamente (no requiere que este en PATH)

set PYTHON=C:\Users\HP\anaconda3\python.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: Verificar que Python existe
if not exist "%PYTHON%" (
    echo ERROR: No se encontro Python en %PYTHON%
    echo Ajusta la variable PYTHON al inicio de este archivo.
    pause
    exit /b 1
)

:: Las variables de entorno las carga el propio analyzer.py desde .env
:: No es necesario leerlas aqui

echo.
echo ========================================
echo   ETF Analyzer  ^|  Dashboard: http://localhost:5010
echo ========================================
echo.

"%PYTHON%" analyzer.py

pause
