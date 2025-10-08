@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ===============================
REM Run IV Agent and publish static
REM files to Vercel public directory.
REM ===============================

REM Jump to this script's folder
cd /d "%~dp0"

REM Output folder served by Vercel
set "OUT_DIR=public\iv-agent"

REM Ensure the folder exists
if not exist "%OUT_DIR%" (
  echo [*] Creating output folder: "%OUT_DIR%"
  mkdir "%OUT_DIR%"
)

echo [*] Installing/Updating minimal dependencies...
python -m pip install --quiet --upgrade requests yfinance

echo [*] Running IV Agent...
python iv_agent.py ^
  --tickers "RELIANCE,ICICIBANK,SBIN,TCS,INFY,ITC,BPCL,AMBUJACEM,AXISBANK,APOLLOHOSP,ASIANPAINT,BAJAJ_AUTO,BAJFINANCE,BEL,BRITANNIA,BSE,DALBHARAT,DIVISLAB,DIXON,EICHERMOT,GRASIM,HAVELLS,HCLTECH,HDFCLIFE,HEROMOTOCO,HINDALCO,ICICIPRULI,INDIGO,INDUSINDBK,JSWSTEEL,JINDALSTEL,JUBLFOOD,KOTAKBANK,LAURUSLABS,LICHSGFIN,LT,LTIM,M&M,ONGC,SHREECEM,TATAMOTORS,TATASTEEL,TECHM,TITAN,UPL,ADANIENT,ADANIPORTS" ^
  --days 30 ^
  --live_iv --use_yfinance_fallback ^
  --top_n 3 ^
  --out_csv "%OUT_DIR%\iv_agent_output.csv" ^
  --html_out "%OUT_DIR%\index.html" ^
  --open_html

if errorlevel 1 (
  echo [x] Failed. Check Python, connectivity (NSE/yfinance), and iv_agent.py
) else (
  echo [✓] Done.
  echo Static site path: %CD%\%OUT_DIR%\index.html
  echo CSV path       : %CD%\%OUT_DIR%\iv_agent_output.csv
  echo ----------------------------------------------------
  echo Deploy notes:
  echo  - Commit both files under public\iv-agent\ to your repo
  echo  - Visit https://<your-project>.vercel.app/iv-agent/
  echo  - CSV at /iv-agent/iv_agent_output.csv
)
pause
