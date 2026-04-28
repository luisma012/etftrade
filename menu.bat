@echo off
chcp 65001 >nul 2>&1
title ETFTRADE - Menu Principal

set PYTHON=C:\Users\HP\anaconda3\python.exe
set PIP=C:\Users\HP\anaconda3\Scripts\pip.exe
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:MENU
cls
echo.
echo  ============================================================
echo    ETFTRADE  ^|  50%% Tecnico + 50%% ML
echo    ETFs: SPY VGT SCHD IFRA SCHF GLD IWM XLE  ^|  Puerto: 5010
echo  ============================================================
echo.
echo    [1]  Iniciar Bot  (app.py - Flask API)
echo    [2]  Iniciar Dashboard Avanzado  (analyzer.py - Dash)
echo    [3]  Correr Backtest del Bot
echo    [4]  Comparar Umbrales de Entrada
echo    [5]  Reentrenar Modelos ML
echo    [6]  Ver Portafolio / P^&L
echo    [7]  Ver Senales Pendientes
echo    [8]  Trigger Scan Manual
echo    [9]  Ver Sentimiento (Fear ^& Greed)
echo    [A]  Auditoria -- Chequeo para pasar a LIVE
echo    [B]  Cerrar posiciones NO-ETF (AMD/AMZN/etc)
echo    [0]  Salir
echo.
set /p OPCION="  Elige una opcion: "

if "%OPCION%"=="1" goto INICIAR_BOT
if "%OPCION%"=="2" goto INICIAR_DASH
if "%OPCION%"=="3" goto BACKTEST
if "%OPCION%"=="4" goto BACKTEST_UMBRALES
if "%OPCION%"=="5" goto REENTRENAR
if "%OPCION%"=="6" goto PORTAFOLIO
if "%OPCION%"=="7" goto PENDIENTES
if "%OPCION%"=="8" goto SCAN
if "%OPCION%"=="9" goto SENTIMIENTO
if /i "%OPCION%"=="A" goto AUDITORIA
if /i "%OPCION%"=="B" goto CERRAR_NO_ETF
if "%OPCION%"=="0" goto SALIR
goto MENU

:INICIAR_BOT
cls
echo.
echo  Iniciando ETF Trader en http://localhost:5010
echo  Presiona Ctrl+C para detener y volver al menu.
echo.
if not exist ".env" (
    echo  [ERROR] Archivo .env no encontrado. Copia .env.template a .env y configura tus credenciales.
    pause
    goto MENU
)
if not exist "ml_models" mkdir ml_models
"%PYTHON%" -X utf8 app.py
pause
goto MENU

:INICIAR_DASH
cls
echo.
echo  Iniciando Dashboard Avanzado en http://localhost:5010
echo  Presiona Ctrl+C para detener y volver al menu.
echo.
"%PYTHON%" analyzer.py
pause
goto MENU

:BACKTEST
cls
echo.
echo  Corriendo backtest del bot (puede tardar ~1 min)...
echo.
"%PYTHON%" -X utf8 backtest_bot.py
echo.
pause
goto MENU

:BACKTEST_UMBRALES
cls
echo.
echo  Comparando umbrales 50 / 55 / 60 / 65 / 70 (puede tardar ~2 min)...
echo.
"%PYTHON%" -X utf8 backtest_thresholds.py
echo.
pause
goto MENU

:REENTRENAR
cls
echo.
echo  Reentrenando modelos ML via API...
echo  (El bot debe estar corriendo en puerto 5010)
echo.
curl -s -X POST http://localhost:5010/api/retrain
echo.
echo.
pause
goto MENU

:PORTAFOLIO
cls
echo.
echo  Portafolio actual:
echo  (El bot debe estar corriendo en puerto 5010)
echo.
curl -s http://localhost:5010/api/portfolio
echo.
echo.
pause
goto MENU

:PENDIENTES
cls
echo.
echo  Senales pendientes de aprobacion:
echo  (El bot debe estar corriendo en puerto 5010)
echo.
curl -s http://localhost:5010/api/pending
echo.
echo.
pause
goto MENU

:SCAN
cls
echo.
echo  Disparando scan manual...
echo  (El bot debe estar corriendo en puerto 5010)
echo.
curl -s -X POST http://localhost:5010/api/scan
echo.
echo.
pause
goto MENU

:SENTIMIENTO
cls
echo.
echo  Fear ^& Greed + SPY RSI actuales:
echo  (El bot debe estar corriendo en puerto 5010)
echo.
curl -s http://localhost:5010/api/sentiment
echo.
echo.
pause
goto MENU

:AUDITORIA
cls
echo.
echo  Corriendo auditoria de criterios para LIVE...
echo.
"%PYTHON%" -X utf8 check_audit.py
echo.
pause
goto MENU

:CERRAR_NO_ETF
cls
echo.
echo  Cerrando posiciones NO-ETF (AMD, AMZN, NVDA, TSLA, etc)...
echo  (El bot debe estar corriendo en puerto 5010)
echo.
echo  Posiciones NO-ETF actuales:
curl -s http://localhost:5010/api/non-etf-positions
echo.
echo.
set /p CONFIRMAR="  Confirmar cierre? (S/N): "
if /i "%CONFIRMAR%"=="S" (
    echo.
    echo  Cerrando...
    curl -s -X POST http://localhost:5010/api/close-non-etf
    echo.
) else (
    echo  Cancelado.
)
echo.
pause
goto MENU

:SALIR
cls
echo.
echo  Hasta luego.
echo.
exit /b 0
