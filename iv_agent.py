#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV Agent for NSE Stocks – CSV + Website (Cards + Click-to-Detail + Payoff Graph)

- Home shows ONLY stock cards (Symbol, Spot, IV, Primary Strategy)
- Click a card → detail view with:
    • Full IC / BPS / LS stats (per share & per 1 lot)
    • Interactive payoff graph @ expiry (toggle strategies; ±1σ/±1.5σ/±2σ range)
    • Math notes + live example
- Lot sizes auto-fetched from NSE (Permitted lot size .csv)
- CSV is UTF-8-BOM for Excel (₹ / ± / – render correctly)

Run (example):
  python iv_agent.py --tickers "RELIANCE,ICICIBANK,HDFCBANK,SBIN,TCS,INFY,ITC,BPCL,AMBUJACEM,AXISBANK,APOLLOHOSP,ASIANPAINT,BAJAJ_AUTO,BAJFINANCE,BEL,BRITANNIA,BSE,DALBHARAT,DIVISLAB,DIXON,EICHERMOT,GRASIM,HAVELLS,HCLTECH,HDFCLIFE,HEROMOTOCO,HINDALCO,ICICIPRULI,INDIGO,INDUSINDBK,JSWSTEEL,JINDALSTEL,JUBLFOOD,KOTAKBANK,LAURUSLABS,LICHSGFIN,LT,LTIM,M&M,ONGC,SHREECEM,TATAMOTORS,TATASTEEL,TECHM,TITAN,UPL,ADANIENT,ADANIPORTS" ^
    --days 30 --live_iv --use_yfinance_fallback ^
    --sigma_ic 1.3 --sigma_bps 1.1 --sigma_ls 0.9 ^
    --width_pct_ic 0.025 --width_pct_bps 0.02 ^
    --default_iv 0.25 --out_csv iv_live_30d.csv --html_out iv_live_30d.html --open_html
"""

from __future__ import annotations
import argparse, csv, math, sys, time, datetime as dt, os, webbrowser, io, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

# --------------------------- Math & Pricing ----------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S: float, K: float, t: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
    S = max(1e-9, float(S)); K = max(1e-9, float(K)); t = max(1e-9, float(t)); sigma = max(1e-9, float(sigma))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return S*math.exp(-q*t)*norm_cdf(d1) - K*math.exp(-r*t)*norm_cdf(d2)
    else:
        return K*math.exp(-r*t)*norm_cdf(-d2) - S*math.exp(-q*t)*norm_cdf(-d1)

def suggest_stride(spot: float) -> int:
    if spot < 200: return 1
    if spot < 500: return 5
    if spot < 1000: return 10
    if spot < 2000: return 20
    if spot < 4000: return 25
    return 50

def round_to_stride(x: float, stride: int) -> int:
    return int(round(x / stride) * stride)

# ---------------------------- Config / Aliases -------------------------------

NSE_SYMBOL_ALIASES = {
    "BAJAJ_AUTO": "BAJAJ-AUTO",
}
def normalize_symbol_for_api(sym: str) -> str:
    s = sym.strip().upper()
    s = NSE_SYMBOL_ALIASES.get(s, s)
    return quote(s, safe="")  # URL-encode (e.g., M&M -> M%26M)

def normalize_symbol_key(sym: str) -> str:
    s = sym.strip().upper()
    return NSE_SYMBOL_ALIASES.get(s, s)

# ------------------------------ Data Models ----------------------------------

@dataclass
class StrategyResult:
    name: str
    action: str
    credit_or_debit: float   # +credit (sell) or -debit (buy) per share
    be_low: Optional[float]
    be_high: Optional[float]
    max_profit: Optional[float]
    max_loss: Optional[float]
    roi: Optional[float]
    strikes: Dict[str, int]

# ------------------------------ NSE Fetch ------------------------------------

NSE_HEADERS = {
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "accept-language": "en-US,en;q=0.9",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _nse_session() -> requests.Session:
    s = requests.Session()
    s.get("https://www.nseindia.com/option-chain", headers=NSE_HEADERS, timeout=10)
    return s

def fetch_iv_spot_from_nse(symbol: str, horizon_days: int, retries: int = 3, sleep_s: float = 1.2
                           ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    try:
        sym_q = normalize_symbol_for_api(symbol)
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={sym_q}"
        sess = _nse_session(); resp = None
        for _ in range(retries):
            resp = sess.get(url, headers={**NSE_HEADERS, "referer": "https://www.nseindia.com/option-chain"}, timeout=12)
            if resp.status_code == 200: break
            time.sleep(sleep_s); sess = _nse_session()
        if resp is None or resp.status_code != 200: return None, None, None
        data = resp.json()
        rec = data.get("records", {})
        underlying = rec.get("underlyingValue")
        if not underlying: return None, None, None
        underlying = float(underlying)

        # choose closest expiry to horizon
        today = dt.date.today(); target = today + dt.timedelta(days=horizon_days)
        exps = rec.get("expiryDates", []) or []
        def p(x):
            try: return dt.datetime.strptime(x, "%d-%b-%Y").date()
            except: return None
        exps = [(s, p(s)) for s in exps]; exps = [(s,d) for s,d in exps if d]
        if not exps: return underlying, None, None
        chosen_exp, _ = min(exps, key=lambda x: abs((x[1]-target).days))

        # filter rows for chosen expiry
        chain = rec.get("data", [])
        rows = [r for r in chain if r.get("expiryDate")==chosen_exp]
        if not rows:
            rows = (data.get("filtered", {}) or {}).get("data", [])
            rows = [r for r in rows if r.get("expiryDate")==chosen_exp]
        if not rows: return underlying, None, chosen_exp

        # closest strike to spot
        atm_row, best = None, 1e18
        for r0 in rows:
            sp = r0.get("strikePrice")
            if sp is None: continue
            d = abs(float(sp)-underlying)
            if d < best: best = d; atm_row = r0
        if not atm_row: return underlying, None, chosen_exp

        ce_iv = atm_row.get("CE", {}).get("impliedVolatility")
        pe_iv = atm_row.get("PE", {}).get("impliedVolatility")
        iv = None
        if ce_iv is not None and pe_iv is not None: iv = (float(ce_iv)+float(pe_iv))/200.0
        elif ce_iv is not None: iv = float(ce_iv)/100.0
        elif pe_iv is not None: iv = float(pe_iv)/100.0

        return underlying, iv, chosen_exp
    except Exception:
        return None, None, None

# ---------- Permitted lot size (official NSE CSV) ----------------------------

LOT_CSV_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

def fetch_lot_sizes() -> Dict[str, int]:
    """Return {SYMBOL: lot} from NSE's fo_mktlots.csv; fallback empty dict."""
    try:
        sess = _nse_session()
        resp = sess.get(LOT_CSV_URL, headers=NSE_HEADERS, timeout=12)
        if resp.status_code != 200:
            time.sleep(1.0)
            sess = _nse_session()
            resp = sess.get(LOT_CSV_URL, headers=NSE_HEADERS, timeout=12)
        if resp.status_code != 200:
            return {}
        rdr = csv.DictReader(io.StringIO(resp.text))
        out: Dict[str,int] = {}
        for row in rdr:
            sym = (row.get("SYMBOL") or row.get("Symbol") or row.get("Underlying") or "").strip().upper()
            lot = (row.get("MARKET LOT") or row.get("MarketLot") or row.get("LOT_SIZE") or
                   row.get("Lot Size") or row.get("Market Lot") or "")
            if not sym or not lot:
                continue
            try:
                out[sym] = int(float(lot))
            except:
                continue
        return out
    except Exception:
        return {}

# ------------------------------ Strategy Engine ------------------------------

@dataclass
class StrategyInputs:
    spot: float
    iv: float
    lot: int
    days: int
    r: float
    q: float
    sigma_ic: float
    sigma_bps: float
    sigma_ls: float
    width_pct_ic: float
    width_pct_bps: float

@dataclass
class BuiltExtras:
    S: float; iv: float; t: float; em: float
    ic_width: int; bps_width: int

def build_strategies(inp: StrategyInputs
                     ) -> Tuple[StrategyResult, StrategyResult, StrategyResult, StrategyResult, BuiltExtras]:
    S = max(1e-9, float(inp.spot))
    iv = max(0.01, min(float(inp.iv), 3.0))
    t = max(1e-9, inp.days/365.0)

    em = S * iv * math.sqrt(t)  # 1σ move
    stride = max(1, suggest_stride(S))
    width_ic = max(stride*2, round_to_stride(inp.width_pct_ic*S, stride))
    width_bps = max(stride*2, round_to_stride(inp.width_pct_bps*S, stride))

    # Iron Condor
    sc = max(round_to_stride(S + inp.sigma_ic*em, stride), stride)
    lc = max(sc + width_ic, stride)
    sp = max(round_to_stride(S - inp.sigma_ic*em, stride), stride)
    lp = max(sp - width_ic, stride)

    sc_p = bs_price(S, sc, t, inp.r, iv, inp.q, "call")
    lc_p = bs_price(S, lc, t, inp.r, iv, inp.q, "call")
    sp_p = bs_price(S, sp, t, inp.r, iv, inp.q, "put")
    lp_p = bs_price(S, lp, t, inp.r, iv, inp.q, "put")

    ic_credit = max(0.0, (sc_p-lc_p) + (sp_p-lp_p))
    ic_width_calc = max(lc - sc, sp - lp)
    ic_risk = max(1e-9, ic_width_calc - ic_credit)
    ic_roi = ic_credit / ic_risk
    ic = StrategyResult("Iron Condor", "Sell premium", ic_credit, sp-ic_credit, sc+ic_credit,
                        ic_credit, ic_risk, ic_roi,
                        {"ShortPut": sp, "LongPut": lp, "ShortCall": sc, "LongCall": lc})

    # Bull Put Spread
    bps_sp = max(round_to_stride(S - inp.sigma_bps*em, stride), stride)
    bps_lp = max(bps_sp - width_bps, stride)
    bps_credit = max(0.0, bs_price(S, bps_sp, t, inp.r, iv, inp.q, "put") - bs_price(S, bps_lp, t, inp.r, iv, inp.q, "put"))
    bps_width = bps_sp - bps_lp
    bps_risk = max(1e-9, bps_width - bps_credit)
    bps_roi = bps_credit / bps_risk
    bps = StrategyResult("Bull Put Spread", "Sell premium", bps_credit, bps_sp-bps_credit, None,
                         bps_credit, bps_risk, bps_roi, {"ShortPut": bps_sp, "LongPut": bps_lp})

    # Long Strangle
    ls_cK = max(round_to_stride(S + inp.sigma_ls*em, stride), stride)
    ls_pK = max(round_to_stride(S - inp.sigma_ls*em, stride), stride)
    ls_debit = bs_price(S, ls_cK, t, inp.r, iv, inp.q, "call") + bs_price(S, ls_pK, t, inp.r, iv, inp.q, "put")
    ls = StrategyResult("Long Strangle", "Buy vol", -ls_debit, ls_pK-ls_debit, ls_cK+ls_debit,
                        None, ls_debit, None, {"Put": ls_pK, "Call": ls_cK})

    # Primary pick
    if iv <= 0.20: primary = ls
    elif iv >= 0.32: primary = bps
    else: primary = ic

    extras = BuiltExtras(S=S, iv=iv, t=t, em=em, ic_width=ic_width_calc, bps_width=bps_width)
    return ic, bps, ls, primary, extras

# ------------------------------ CSV + HTML -----------------------------------

def write_csv(out_path: str, rows: List[Dict[str, str]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM for Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# Use SAFE tokens instead of str.format braces to avoid conflicts with CSS/JS
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IV Agent</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{
    --bg:#070b16; --card:#0f1629; --panel:#0b1222;
    --txt:#e8ecf6; --muted:#a4afc6; --line:#1b2a4a; --accent:#7aa2f7;
    --ok:#34d399; --warn:#f59e0b; --bad:#ef4444;
    --ic:#22d3ee; --bps:#a7f3d0; --ls:#f9a8d4;
  }
  *{box-sizing:border-box}
  body{margin:0;background:
    radial-gradient(1000px 600px at 10% -10%, #1c2550 0%, transparent 60%),
    radial-gradient(900px 600px at 110% 10%, #1d3a3f 0%, transparent 55%),
    var(--bg);
    color:var(--txt); font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Ubuntu,Arial,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:24px}
  h1{margin:0 0 4px 0;font-size:28px;letter-spacing:.3px}
  .meta{color:var(--muted);margin-bottom:18px}
  .toolbar{display:flex;gap:10px;align-items:center;margin:6px 0 20px 0}
  .search{flex:1;max-width:420px;background:#0b1427;border:1px solid #1d2a49;color:var(--txt);
          border-radius:12px;padding:10px 12px;outline:none}
  .btn{background:var(--accent);color:#fff;border:none;border-radius:12px;padding:10px 14px;cursor:pointer;text-decoration:none}
  .btn.secondary{background:#203055;color:#dbe4ff}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
  .card{background:linear-gradient(180deg,#111b33 0%,#0d1428 100%);border:1px solid #1a2747;
        border-radius:16px;padding:14px;cursor:pointer;transition:transform .12s ease, border-color .12s}
  .card:hover{transform:translateY(-2px);border-color:#2a3a6a}
  .sym{font-weight:700;font-size:18px;letter-spacing:.4px}
  .row{display:flex;justify-content:space-between;margin-top:8px;color:var(--muted)}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-top:10px}
  .pill.sell{background:#173225;color:#8bffa9}
  .pill.buy{background:#2d1f2d;color:#ff9bd1}
  .detail{
    position:fixed;inset:0;background:rgba(5,8,15,.65);
    display:none;align-items:flex-start;justify-content:center;padding:32px 16px;backdrop-filter:blur(4px)
  }
  .panel{width:min(1100px,96vw);max-height:92vh;overflow:auto;background:var(--panel);border:1px solid #1c294b;border-radius:18px;padding:20px}
  .panelhead{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:12px}
  .title{font-size:22px;font-weight:700}
  .sub{color:var(--muted)}
  .tag{padding:2px 8px;border-radius:999px;background:#1c2747;color:#c5d0e8;font-size:12px;margin-left:6px}
  .close{background:#25365e;border:1px solid #334876;color:#dbe4ff;border-radius:10px;padding:8px 10px;cursor:pointer}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .box{background:#0e1933;border:1px solid #1c2a51;border-radius:14px;padding:12px}
  .box h3{margin:0 0 8px 0;font-size:16px}
  .grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
  .row2{display:flex;justify-content:space-between;border-bottom:1px dashed #1c2a51;padding:6px 0}
  canvas{width:100%;height:360px;background:#0b1427;border:1px solid #1b2a4a;border-radius:14px}
  .legend{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}
  .chip{display:flex;align-items:center;gap:6px;background:#162243;border:1px solid #233463;border-radius:999px;padding:6px 10px;cursor:pointer}
  .dot{width:10px;height:10px;border-radius:50%}
  .dot.ic{background:var(--ic)} .dot.bps{background:var(--bps)} .dot.ls{background:var(--ls)}
  .chip.active{outline:2px solid #2f4aa8}
  .range{display:flex;gap:8px;margin-top:6px}
  .range .chip{background:#14203f}
  .muted{color:var(--muted)}
  @media (max-width:900px){ .cols{grid-template-columns:1fr} }
</style>
</head>
<body>
<div class="wrap">
  <h1>IV Agent</h1>
  <div class="meta">Generated: __GENERATED__ · Horizon: __DAYS__ days · Source: NSE · Lot P&L assumes <b>1 lot</b></div>
  <div class="toolbar">
    <input id="q" class="search" placeholder="Search symbol…" />
    <a class="btn" href="__CSV_NAME__" download>Download CSV</a>
  </div>

  <div id="cards" class="grid"></div>
</div>

<!-- Detail Overlay -->
<div id="detail" class="detail" role="dialog" aria-modal="true">
  <div class="panel">
    <div class="panelhead">
      <div>
        <div class="title" id="d_sym">—</div>
        <div class="sub" id="d_meta">—</div>
      </div>
      <button class="close" onclick="closeDetail()">Close</button>
    </div>

    <div class="cols">
      <div class="box">
        <h3>Payoff @ Expiry (₹ per <b>lot</b>)</h3>
        <canvas id="chart" width="900" height="360"></canvas>
        <div class="legend">
          <div id="selIC"  class="chip"><span class="dot ic"></span>Iron Condor</div>
          <div id="selBPS" class="chip"><span class="dot bps"></span>Bull Put Spread</div>
          <div id="selLS"  class="chip"><span class="dot ls"></span>Long Strangle</div>
        </div>
        <div class="range">
          <div id="r1"  class="chip">±1σ</div>
          <div id="r15" class="chip">±1.5σ</div>
          <div id="r2"  class="chip">±2σ</div>
        </div>
        <div class="muted" id="explain" style="margin-top:8px"></div>
      </div>

      <div class="box">
        <h3>Strategy Details (per share & per 1 lot)</h3>
        <div class="grid2">
          <div>
            <div class="row2"><span><b>Primary</b></span><span id="primName">—</span></div>
            <div class="row2"><span>Action</span><span id="primAction">—</span></div>
            <div class="row2"><span>Max P/L (₹/sh)</span><span id="primPLsh">—</span></div>
            <div class="row2"><span>Max P/L (₹/lot)</span><span id="primPLlot">—</span></div>
            <div class="row2"><span>Breakevens</span><span id="primBE">—</span></div>
          </div>
          <div id="legs"></div>
        </div>
        <div class="muted" style="margin-top:10px">
          Pricing uses Black–Scholes (European) with continuous dividend yield. IV is annualized (decimal).
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const GLOBAL_DAYS = __DAYS__;
const CURRENCY = "₹";

// Build cards
const cards = document.getElementById('cards');
function pillClass(action){ return action.includes('Sell') ? 'pill sell' : 'pill buy'; }
function card(sym, spot, iv, primary, i) {
  const div = document.createElement('div');
  div.className='card';
  div.onclick=()=>openDetail(i);
  div.innerHTML = `
    <div class="sym">${sym}</div>
    <div class="row"><span>Spot</span><span>₹${spot.toFixed(2)}</span></div>
    <div class="row"><span>IV</span><span>${(iv*100).toFixed(1)}%</span></div>
    <div class="${pillClass(primary.action)}">${primary.name} – ${primary.action}</div>
  `;
  return div;
}
DATA.forEach((d,i)=>cards.appendChild(card(d.symbol,d.spot,d.iv,d.primary,i)));

// Search
document.getElementById('q').addEventListener('input', e=>{
  const t = e.target.value.toLowerCase();
  Array.from(cards.children).forEach(c=>{
    c.style.display = c.textContent.toLowerCase().includes(t) ? '' : 'none';
  });
});

// Detail logic
const detail = document.getElementById('detail');
const d_sym = document.getElementById('d_sym');
const d_meta = document.getElementById('d_meta');
const primName = document.getElementById('primName');
const primAction = document.getElementById('primAction');
const primPLsh = document.getElementById('primPLsh');
const primPLlot = document.getElementById('primPLlot');
const primBE = document.getElementById('primBE');
const legs = document.getElementById('legs');
const explain = document.getElementById('explain');

let CUR = null;
let showIC=true, showBPS=false, showLS=false;
let sigMult = 1.0;

function fmt(x){ return (x===null||x===undefined) ? '—' : (typeof x==='number'? x.toFixed(2): x); }
function openDetail(i){
  CUR = DATA[i];
  d_sym.textContent = CUR.symbol;
  d_meta.textContent = `Expiry: ${CUR.expiry || '—'} · Lot: ${CUR.lot} · Spot: ₹${CUR.spot.toFixed(2)} · IV: ${(CUR.iv*100).toFixed(1)}%`;
  primName.textContent = CUR.primary.name;
  primAction.textContent = CUR.primary.action;
  primPLsh.textContent = `${CUR.primary.max_profit_sh} / ${CUR.primary.max_loss_sh}`;
  primPLlot.textContent = `${CUR.primary.max_profit_lot} / ${CUR.primary.max_loss_lot}`;
  primBE.textContent = `${CUR.primary.be_low?.toFixed(2) ?? '—'}  /  ${CUR.primary.be_high?.toFixed(2) ?? '—'}`;

  // Legs summary
  legs.innerHTML = `
    <div>
      <div class="row2"><span><b>IC</b> Short Put</span><span>${CUR.ic.sp}</span></div>
      <div class="row2"><span>IC Long Put</span><span>${CUR.ic.lp}</span></div>
      <div class="row2"><span>IC Short Call</span><span>${CUR.ic.sc}</span></div>
      <div class="row2"><span>IC Long Call</span><span>${CUR.ic.lc}</span></div>
      <div class="row2"><span>IC Credit (₹/sh · /lot)</span><span>${CUR.ic.credit_sh.toFixed(2)} · ${CUR.ic.credit_lot.toFixed(2)}</span></div>
    </div>
    <div>
      <div class="row2"><span><b>BPS</b> Short Put</span><span>${CUR.bps.sp}</span></div>
      <div class="row2"><span>BPS Long Put</span><span>${CUR.bps.lp}</span></div>
      <div class="row2"><span>BPS Credit (₹/sh · /lot)</span><span>${CUR.bps.credit_sh.toFixed(2)} · ${CUR.bps.credit_lot.toFixed(2)}</span></div>
      <div class="row2"><span><b>LS</b> Put</span><span>${CUR.ls.kp}</span></div>
      <div class="row2"><span>LS Call</span><span>${CUR.ls.kc}</span></div>
      <div class="row2"><span>LS Cost (₹/sh · /lot)</span><span>${CUR.ls.debit_sh.toFixed(2)} · ${CUR.ls.debit_lot.toFixed(2)}</span></div>
    </div>
  `;

  // Default toggles: primary only
  showIC = CUR.primary.name==='Iron Condor';
  showBPS= CUR.primary.name==='Bull Put Spread';
  showLS = CUR.primary.name==='Long Strangle';
  sigMult = 1.0;
  syncToggles();
  draw();
  detail.style.display='flex';
}
function closeDetail(){ detail.style.display='none'; }

// Toggle chips
function syncToggles(){
  const t=(id,on)=>{ const el=document.getElementById(id); el.classList.toggle('active', on); }
  t('selIC',showIC); t('selBPS',showBPS); t('selLS',showLS);
  t('r1',sigMult===1.0); t('r15',sigMult===1.5); t('r2',sigMult===2.0);
}
document.getElementById('selIC').onclick = ()=>{ showIC=!showIC; syncToggles(); draw(); }
document.getElementById('selBPS').onclick= ()=>{ showBPS=!showBPS; syncToggles(); draw(); }
document.getElementById('selLS').onclick = ()=>{ showLS=!showLS; syncToggles(); draw(); }
document.getElementById('r1').onclick  = ()=>{ sigMult=1.0; syncToggles(); draw(); }
document.getElementById('r15').onclick = ()=>{ sigMult=1.5; syncToggles(); draw(); }
document.getElementById('r2').onclick  = ()=>{ sigMult=2.0; syncToggles(); draw(); }

// Payoff (per LOT)
function payoffIC(ST,d){
  const L = d.lot, credit = d.ic.credit_sh;
  const sp=d.ic.sp, lp=d.ic.lp, sc=d.ic.sc, lc=d.ic.lc;
  let callLoss = 0;
  if (ST>sc && ST<lc) callLoss = ST - sc;
  else if (ST>=lc)    callLoss = (lc - sc);
  let putLoss = 0;
  if (ST<sp && ST>lp) putLoss = sp - ST;
  else if (ST<=lp)    putLoss = (sp - lp);
  const perShare = credit - callLoss - putLoss;
  return perShare * L;
}
function payoffBPS(ST,d){
  const L=d.lot, credit=d.bps.credit_sh, sp=d.bps.sp, lp=d.bps.lp, width=(sp-lp);
  let loss = 0;
  if (ST<sp && ST>lp) loss = sp - ST;
  else if (ST<=lp)    loss = width;
  return (credit - loss)*L;
}
function payoffLS(ST,d){
  const L=d.lot, debit=d.ls.debit_sh, kp=d.ls.kp, kc=d.ls.kc;
  const put  = Math.max(0, kp - ST);
  const call = Math.max(0, ST - kc);
  return (put + call - debit) * L;
}

// Drawing
function draw(){
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);

  const d = CUR;
  const em = d.em;
  const S = d.spot;
  const range = sigMult * em;
  const xMin = Math.max(0, S - range*1.05);
  const xMax = S + range*1.05;

  // Build series
  const N=350, xs=[], ysIC=[], ysBPS=[], ysLS=[];
  for(let i=0;i<=N;i++){
    const ST = xMin + (xMax-xMin)*i/N;
    xs.push(ST);
    if(showIC)  ysIC.push(payoffIC(ST,d));
    if(showBPS) ysBPS.push(payoffBPS(ST,d));
    if(showLS)  ysLS.push(payoffLS(ST,d));
  }
  let yMin=0, yMax=0;
  const allY=[...ysIC,...ysBPS,...ysLS];
  if(allY.length){ yMin=Math.min(...allY); yMax=Math.max(...allY); }
  if(yMin===yMax){ yMin-=1; yMax+=1; }
  const pad = (yMax - yMin)*0.12; yMin-=pad; yMax+=pad;

  // helpers
  const X= x=> ( (x-xMin)/(xMax-xMin) )*(c.width-60) + 40;
  const Y= y=> c.height-30 - ((y - yMin)/(yMax - yMin))*(c.height-60);

  // axes
  ctx.strokeStyle="#1b2a4a"; ctx.lineWidth=1;
  ctx.strokeRect(40,20,c.width-80,c.height-50);
  const xS = X(S); ctx.beginPath(); ctx.moveTo(xS,20); ctx.lineTo(xS,c.height-30); ctx.stroke();
  const y0 = Y(0); ctx.beginPath(); ctx.moveTo(40,y0); ctx.lineTo(c.width-40,y0); ctx.stroke();

  // labels
  ctx.fillStyle="#a4afc6"; ctx.font="12px Segoe UI, Roboto, sans-serif";
  ctx.fillText(`S=₹${S.toFixed(2)}`, xS+4, 30);
  ctx.fillText("P/L (₹ per lot)", 46, 32);
  ctx.fillText(`Price @ Expiry`, c.width-150, c.height-12);

  // series
  function plot(ys,color){
    if(!ys.length) return;
    ctx.beginPath();
    ctx.strokeStyle=color; ctx.lineWidth=2;
    for(let i=0;i<=N;i++){
      const px=X(xs[i]), py=Y(ys[i]);
      if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
    }
    ctx.stroke();
  }
  plot(ysIC,  getComputedStyle(document.documentElement).getPropertyValue('--ic').trim());
  plot(ysBPS, getComputedStyle(document.documentElement).getPropertyValue('--bps').trim());
  plot(ysLS,  getComputedStyle(document.documentElement).getPropertyValue('--ls').trim());

  explain.innerHTML = `Range: ₹${(S-range).toFixed(2)} – ₹${(S+range).toFixed(2)} (±${sigMult}σ ≈ ₹${(sigMult*em).toFixed(2)}) · Lot=${d.lot}`;
}
</script>
</body>
</html>
"""

def write_html(out_html: str, csv_name: str, days: int, js_rows: List[dict]) -> None:
    # Safe token replacement (no str.format) to avoid brace conflicts
    html = (HTML
            .replace("__GENERATED__", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            .replace("__DAYS__", str(days))
            .replace("__CSV_NAME__", os.path.basename(csv_name))
            .replace("__DATA_JSON__", json.dumps(js_rows, ensure_ascii=False)))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

# ------------------------------- CSV helpers ---------------------------------

def read_csv_map(path: str, key_col: str, val_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            k = (row.get(key_col) or row.get(key_col.upper()) or "").strip().upper()
            v = row.get(val_col) or row.get(val_col.upper())
            if k and v not in (None, ""):
                try: out[k] = float(v)
                except: pass
    return out

# ------------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="IV Agent – NSE live IV, CSV + Website (cards + detail + payoff)")
    ap.add_argument("--tickers", required=True, type=str, help="Comma-separated NSE symbols")
    ap.add_argument("--days", type=int, default=30, help="Horizon in days")
    ap.add_argument("--risk_free", type=float, default=0.07)
    ap.add_argument("--yield_div", type=float, default=0.00)
    # sigma + widths
    ap.add_argument("--sigma_ic", type=float, default=1.2)
    ap.add_argument("--sigma_bps", type=float, default=1.0)
    ap.add_argument("--sigma_ls", type=float, default=1.0)
    ap.add_argument("--width_pct_ic", type=float, default=0.02)
    ap.add_argument("--width_pct_bps", type=float, default=0.02)
    # data sources
    ap.add_argument("--default_iv", type=float, default=0.25)
    ap.add_argument("--iv_csv", type=str, default="")
    ap.add_argument("--spot_csv", type=str, default="")
    ap.add_argument("--live_iv", action="store_true")
    ap.add_argument("--use_yfinance_fallback", action="store_true")
    # outputs
    ap.add_argument("--out_csv", type=str, default="iv_agent_output.csv")
    ap.add_argument("--html_out", type=str, default="iv_agent_output.html")  # << fixed (type=str)
    ap.add_argument("--open_html", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers: print("No tickers."); sys.exit(1)

    # --- Fetch lot sizes (official file) ---
    lot_map = fetch_lot_sizes()
    if args.verbose:
        print(f"Loaded {len(lot_map)} lot sizes from NSE file.")

    # CSV overrides (if any)
    spot_map = read_csv_map(args.spot_csv, "ticker", "spot") if args.spot_csv else {}
    iv_map   = read_csv_map(args.iv_csv,   "ticker", "iv")   if args.iv_csv   else {}

    csv_rows: List[Dict[str,str]] = []
    js_rows: List[dict] = []
    skipped: List[str] = []

    for sym in tickers:
        key = normalize_symbol_key(sym)
        lot = lot_map.get(key)
        if lot is None:
            lot = 1  # default if not in F&O file

        spot = spot_map.get(sym)
        iv   = iv_map.get(sym)
        chosen_exp = ""

        if args.live_iv:
            s_live, iv_live, chosen_exp_live = fetch_iv_spot_from_nse(sym, args.days)
            if args.verbose: print(f"[NSE] {sym}: spot={s_live} iv={iv_live} exp={chosen_exp_live}")
            if s_live is not None: spot = s_live
            if iv_live is not None: iv   = iv_live
            if chosen_exp_live:     chosen_exp = chosen_exp_live

        if spot is None and args.use_yfinance_fallback:
            try:
                import yfinance as yf
                s = sym if sym.endswith(".NS") else f"{sym}.NS"
                data = yf.Ticker(s).history(period="1d")
                if data is not None and not data.empty:
                    spot = float(data["Close"].iloc[-1])
            except Exception:
                pass
            if args.verbose: print(f"[YF ] {sym}: spot={spot}")

        if iv is None: iv = args.default_iv
        if iv > 1.0: iv = iv / 100.0
        iv = max(0.01, min(iv, 3.0))

        if spot is None:
            if args.verbose: print(f"[SKIP] {sym}: no spot from NSE/CSV{'/yfinance' if args.use_yfinance_fallback else ''}")
            skipped.append(sym); continue

        ic, bps, ls, primary, extra = build_strategies(
            StrategyInputs(
                spot=spot, iv=iv, lot=lot, days=args.days, r=args.risk_free, q=args.yield_div,
                sigma_ic=args.sigma_ic, sigma_bps=args.sigma_bps, sigma_ls=args.sigma_ls,
                width_pct_ic=args.width_pct_ic, width_pct_bps=args.width_pct_bps
            )
        )

        # ----- CSV row (strings for Excel) -----
        def lotify(x: Optional[float]) -> str:
            if x is None: return ""
            return f"{x*lot:.2f}"

        if primary.name == "Long Strangle":
            prim_profit_sh, prim_loss_sh = "Uncapped", f"{ls.max_loss:.2f}"
        else:
            prim_profit_sh, prim_loss_sh = f"{primary.max_profit:.2f}", f"{primary.max_loss:.2f}"

        csv_rows.append({
            "Symbol": sym,
            "Expiry(closest)": chosen_exp,
            "Spot": f"{spot:.2f}",
            "ATM IV": f"{iv:.4f}",
            "Lot": lot,
            "1σ Move(₹)": f"{extra.em:.2f}",
            "Range ±1σ": f"{spot - extra.em:.2f}–{spot + extra.em:.2f}",
            # Primary
            "Primary Strategy": primary.name,
            "Action (Buy/Sell)": primary.action,
            "Primary Max Profit (₹/sh)": prim_profit_sh,
            "Primary Max Loss (₹/sh)":   prim_loss_sh,
            "Primary Max Profit (₹/lot)": "Uncapped" if primary.name=="Long Strangle" else lotify(primary.max_profit),
            "Primary Max Loss (₹/lot)":   lotify(primary.max_loss),
            # IC
            "IC Short Put":  ic.strikes["ShortPut"],
            "IC Long Put":   ic.strikes["LongPut"],
            "IC Short Call": ic.strikes["ShortCall"],
            "IC Long Call":  ic.strikes["LongCall"],
            "IC Credit (₹/sh)": f"{ic.credit_or_debit:.2f}",
            "IC Credit (₹/lot)": lotify(ic.credit_or_debit),
            "IC BE Low": f"{ic.be_low:.2f}",
            "IC BE High": f"{ic.be_high:.2f}",
            "IC Max Risk (₹/sh)": f"{ic.max_loss:.2f}",
            "IC Max Risk (₹/lot)": lotify(ic.max_loss),
            "IC ROI %": f"{ic.roi*100:.1f}",
            # BPS
            "BPS Short Put": bps.strikes["ShortPut"],
            "BPS Long Put":  bps.strikes["LongPut"],
            "BPS Credit (₹/sh)": f"{bps.credit_or_debit:.2f}",
            "BPS Credit (₹/lot)": lotify(bps.credit_or_debit),
            "BPS BE": f"{bps.be_low:.2f}",
            "BPS Max Risk (₹/sh)": f"{bps.max_loss:.2f}",
            "BPS Max Risk (₹/lot)": lotify(bps.max_loss),
            "BPS ROI %": f"{bps.roi*100:.1f}",
            # LS
            "LS Put":  ls.strikes["Put"],
            "LS Call": ls.strikes["Call"],
            "LS Cost (₹/sh)": f"{-ls.credit_or_debit:.2f}",
            "LS Cost (₹/lot)": lotify(-ls.credit_or_debit),
            "LS BE Low": f"{ls.be_low:.2f}",
            "LS BE High": f"{ls.be_high:.2f}",
        })

        # ----- JS (numeric) row for website -----
        js_rows.append({
            "symbol": sym,
            "expiry": chosen_exp,
            "spot": float(f"{spot:.6f}"),
            "iv": float(f"{iv:.6f}"),
            "lot": int(lot),
            "days": args.days,
            "em": float(f"{extra.em:.6f}"),
            "primary": {
                "name": primary.name,
                "action": primary.action,
                "be_low": None if primary.be_low is None else float(f"{primary.be_low:.6f}"),
                "be_high": None if primary.be_high is None else float(f"{primary.be_high:.6f}"),
                "max_profit_sh": "Uncapped" if primary.name=="Long Strangle" else f"{primary.max_profit:.2f}",
                "max_loss_sh": f"{primary.max_loss:.2f}" if isinstance(primary.max_loss,(int,float)) else "—",
                "max_profit_lot": "Uncapped" if primary.name=="Long Strangle" else f"{(primary.max_profit or 0)*lot:.2f}",
                "max_loss_lot": f"{(primary.max_loss or 0)*lot:.2f}",
            },
            "ic": {
                "sp": ic.strikes["ShortPut"], "lp": ic.strikes["LongPut"],
                "sc": ic.strikes["ShortCall"], "lc": ic.strikes["LongCall"],
                "credit_sh": float(f"{ic.credit_or_debit:.6f}"),
                "credit_lot": float(f"{ic.credit_or_debit*lot:.6f}"),
                "width": int(extra.ic_width),
                "be_low": float(f"{ic.be_low:.6f}"), "be_high": float(f"{ic.be_high:.6f}"),
                "risk_sh": float(f"{ic.max_loss:.6f}"), "roi": float(f"{ic.roi:.6f}")
            },
            "bps": {
                "sp": bps.strikes["ShortPut"], "lp": bps.strikes["LongPut"],
                "credit_sh": float(f"{bps.credit_or_debit:.6f}"),
                "credit_lot": float(f"{bps.credit_or_debit*lot:.6f}"),
                "be": float(f"{bps.be_low:.6f}"),
                "risk_sh": float(f"{bps.max_loss:.6f}"),
                "width": int(extra.bps_width),
                "roi": float(f"{bps.roi:.6f}")
            },
            "ls": {
                "kp": ls.strikes["Put"], "kc": ls.strikes["Call"],
                "debit_sh": float(f"{-ls.credit_or_debit:.6f}"),
                "debit_lot": float(f"{-ls.credit_or_debit*lot:.6f}"),
                "be_low": float(f"{ls.be_low:.6f}"), "be_high": float(f"{ls.be_high:.6f}")
            }
        })

    if not csv_rows:
        print("No results (every ticker skipped)."); sys.exit(2)

    write_csv(args.out_csv, csv_rows)
    write_html(args.html_out, args.out_csv, args.days, js_rows)

    print(f"\nSaved CSV : {os.path.abspath(args.out_csv)}")
    print(f"Saved HTML: {os.path.abspath(args.html_out)}")
    if skipped: print("Skipped (no spot): " + ", ".join(skipped))

    if args.open_html:
        webbrowser.open("file://" + os.path.abspath(args.html_out))

if __name__ == "__main__":
    main()
