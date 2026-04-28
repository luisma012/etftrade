@echo off
REM ============================================================
REM  ETF Trader — No LLM — Flask port 5010
REM  Usa Anaconda: C:\Users\HP\anaconda3\
REM ============================================================

title ETF Trader (No LLM) — Port 5010

set PYTHON=C:\Users\HP\anaconda3\python.exe
set PIP=C:\Users\HP\anaconda3\Scripts\pip.exe

echo.
echo  ================================================
echo   ETF TRADER  ^|  50%% Technical + 50%% ML
echo   Brokers  : Robinhood / Alpaca
echo   ETFs     : SPY, VGT, SCHD, IFRA, SCHF
echo   Sentiment: Fear ^& Greed + SPY RSI
echo   Port     : 5010
echo   Python   : %PYTHON%
echo  ================================================
echo.

REM ── Verificar que existe Anaconda ────────────────────────────
if not exist "%PYTHON%" (
    echo [ERROR] No se encontro Python en %PYTHON%
    echo         Verifica la ruta de Anaconda.
    pause
    exit /b 1
)

REM ── Check .env ────────────────────────────────────────────────
if not exist ".env" (
    echo [WARN] .env no encontrado. Copiando desde .env.template...
    copy .env.template .env >nul
    echo [WARN] Edita .env con tus credenciales y vuelve a ejecutar.
    pause
    exit /b 1
)

REM ── Crear directorio de modelos ML ───────────────────────────
if not exist "ml_models" mkdir ml_models

REM ── Instalar dependencias ────────────────────────────────────
echo [INFO] Verificando dependencias...
"%PIP%" install -r requirements.txt --quiet

REM ── Lanzar ───────────────────────────────────────────────────
echo [INFO] Iniciando ETF Trader en http://localhost:5010
echo [INFO] Presiona Ctrl+C para detener.
echo.
"%PYTHON%" -X utf8 app.py

pause
