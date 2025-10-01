@echo off
setlocal

REM === Paths (keep the quotes) ===
set PYEXE="C:\Users\arjun\Desktop\Arjun\Algo Trading\NEW BOT\Crypto-Trading-Bot-master\Crypto-Trading-Bot-master\.conda\python.exe"
set SCRIPT="C:\Users\arjun\Desktop\Arjun\Algo Trading\Funda\iv_agent.py"
set OUTCSV="C:\Users\arjun\Desktop\Arjun\Algo Trading\Funda\iv_live_30d.csv"

echo.
echo Installing dependencies (pip, requests) into your chosen Python...
%PYEXE% -m pip install --upgrade pip requests

echo.
echo Running IV Agent with live IV+spot from NSE...
%PYEXE% %SCRIPT% ^
  --tickers "RELIANCE,ICICIBANK,HDFCBANK,SBIN,TCS,INFY,ITC,BPCL,AMBUJACEM,AXISBANK,APOLLOHOSP,ASIANPAINT,BAJAJ_AUTO,BAJFINANCE,BEL,BRITANNIA,BSE,DALBHARAT,DIVISLAB,DIXON,EICHERMOT,GRASIM,HAVELLS,HCLTECH,HDFCLIFE,HEROMOTOCO,HINDALCO,ICICIPRULI,INDIGO,INDUSINDBK,JSWSTEEL,JINDALSTEL,JUBLFOOD,KOTAKBANK,LAURUSLABS,LICHSGFIN,LT,LTIM,M&M,ONGC,SHREECEM,TATAMOTORS,TATASTEEL,TECHM,TITAN,UPL,ADANIENT,ADANIPORTS" ^
  --days 30 ^
  --live_iv ^
  --sigma_ic 1.3 ^
  --sigma_bps 1.1 ^
  --sigma_ls 0.9 ^
  --width_pct_ic 0.025 ^
  --width_pct_bps 0.02 ^
  --out_csv %OUTCSV%

echo.
echo Done! CSV saved to:
echo %OUTCSV%
echo Open it in Excel to see strategies, strikes, breakevens, ROI, and P&L.
echo.
pause
endlocal
