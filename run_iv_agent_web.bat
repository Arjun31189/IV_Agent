@echo off
cd /d "%~dp0"

echo [*] Installing/Updating minimal dependencies...
python -m pip install --quiet --upgrade requests yfinance

echo [*] Running IV Agent...
python iv_agent.py ^
  --tickers "RELIANCE,ICICIBANK,SBIN,TCS,INFY,ITC,BPCL,AMBUJACEM,AXISBANK,APOLLOHOSP,ASIANPAINT,BAJAJ_AUTO,BAJFINANCE,BEL,BRITANNIA,BSE,DALBHARAT,DIVISLAB,DIXON,EICHERMOT,GRASIM,HAVELLS,HCLTECH,HDFCLIFE,HEROMOTOCO,HINDALCO,ICICIPRULI,INDIGO,INDUSINDBK,JSWSTEEL,JINDALSTEL,JUBLFOOD,KOTAKBANK,LAURUSLABS,LICHSGFIN,LT,LTIM,M&M,ONGC,SHREECEM,TATAMOTORS,TATASTEEL,TECHM,TITAN,UPL,ADANIENT,ADANIPORTS" ^
  --days 30 ^
  --live_iv --use_yfinance_fallback ^
  --top_n 3 ^
  --out_csv "iv_agent_output.csv" ^
  --html_out "iv_agent_output.html" ^
  --open_html

if errorlevel 1 (
  echo [x] Failed. Check Python and iv_agent.py
) else (
  echo [✓] Done.
  echo CSV : %CD%\iv_agent_output.csv
  echo HTML: %CD%\iv_agent_output.html
)
pause
