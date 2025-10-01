@echo off
setlocal
set PYEXE="C:\Users\arjun\Desktop\Arjun\Algo Trading\NEW BOT\Crypto-Trading-Bot-master\Crypto-Trading-Bot-master\.conda\python.exe"
set SCRIPT="C:\Users\arjun\Desktop\Arjun\Algo Trading\Funda\iv_agent.py"
set OUTCSV="C:\Users\arjun\Desktop\Arjun\Algo Trading\Funda\iv_live_30d.csv"
set OUTHTML="C:\Users\arjun\Desktop\Arjun\Algo Trading\Funda\iv_live_30d.html"

echo Installing/Updating deps...
%PYEXE% -m pip install --upgrade pip requests

echo Running IV Agent (live IV + website)...
%PYEXE% %SCRIPT% ^
  --tickers "RELIANCE,ICICIBANK,HDFCBANK,SBIN,TCS,INFY,ITC,BPCL,AMBUJACEM,AXISBANK,APOLLOHOSP,ASIANPAINT,BAJAJ_AUTO,BAJFINANCE,BEL,BRITANNIA,BSE,DALBHARAT,DIVISLAB,DIXON,EICHERMOT,GRASIM,HAVELLS,HCLTECH,HDFCLIFE,HEROMOTOCO,HINDALCO,ICICIPRULI,INDIGO,INDUSINDBK,JSWSTEEL,JINDALSTEL,JUBLFOOD,KOTAKBANK,LAURUSLABS,LICHSGFIN,LT,LTIM,M&M,ONGC,SHREECEM,TATAMOTORS,TATASTEEL,TECHM,TITAN,UPL,ADANIENT,ADANIPORTS" ^
  --days 30 ^
  --live_iv --use_yfinance_fallback ^
  --sigma_ic 1.3 --sigma_bps 1.1 --sigma_ls 0.9 ^
  --width_pct_ic 0.025 --width_pct_bps 0.02 ^
  --default_iv 0.25 ^
  --out_csv %OUTCSV% ^
  --html_out %OUTHTML% ^
  --open_html

echo Done!
echo CSV : %OUTCSV%
echo HTML: %OUTHTML%
pause
endlocal
