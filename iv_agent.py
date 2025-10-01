#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV Agent – NSE live IV → Multi-strategy ranking + Website (Top Picks + Payoff)

Adds & ranks (most) strategies you listed. Naked shorts are OFF by default.
Calendars/Jade-Lizard/Batman/Double-Plateau/Range-Forward are stubbed.

Run (example):
  python iv_agent.py --tickers "RELIANCE,ICICIBANK,SBIN,TCS,INFY,ITC,BPCL,AMBUJACEM,AXISBANK,APOLLOHOSP,ASIANPAINT,BAJAJ_AUTO,BAJFINANCE,BEL,BRITANNIA,BSE,DALBHARAT,DIVISLAB,DIXON,EICHERMOT,GRASIM,HAVELLS,HCLTECH,HDFCLIFE,HEROMOTOCO,HINDALCO,ICICIPRULI,INDIGO,INDUSINDBK,JSWSTEEL,JINDALSTEL,JUBLFOOD,KOTAKBANK,LAURUSLABS,LICHSGFIN,LT,LTIM,M&M,ONGC,SHREECEM,TATAMOTORS,TATASTEEL,TECHM,TITAN,UPL,ADANIENT,ADANIPORTS" ^
    --days 30 --live_iv --use_yfinance_fallback --open_html ^
    --sigma_ic 1.3 --sigma_bps 1.1 --sigma_ls 0.9 ^
    --width_pct_ic 0.025 --width_pct_bps 0.02 ^
    --default_iv 0.25 --top_n 3

Safety:
  Add --allow_naked if you want naked Sell Call / Sell Put / Short Strangle / Short Straddle considered.
"""

from __future__ import annotations
import argparse, csv, math, sys, time, datetime as dt, os, webbrowser, io, json, random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import requests

# --------------------------- Math & Pricing ----------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    invsqrt2pi = 0.3989422804014327
    return invsqrt2pi * math.exp(-0.5 * x * x)

def bs_price(S: float, K: float, t: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
    S = max(1e-9, float(S)); K = max(1e-9, float(K)); t = max(1e-9, float(t)); sigma = max(1e-6, float(sigma))
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

NSE_SYMBOL_ALIASES = { "BAJAJ_AUTO": "BAJAJ-AUTO" }
def normalize_symbol_for_api(sym: str) -> str:
    s = sym.strip().upper(); s = NSE_SYMBOL_ALIASES.get(s, s)
    return quote(s, safe="")
def normalize_symbol_key(sym: str) -> str:
    s = sym.strip().upper(); return NSE_SYMBOL_ALIASES.get(s, s)

# ------------------------------ Data Models ----------------------------------

@dataclass
class Strategy:
    key: str
    name: str
    style: str    # 'credit' or 'debit'
    risk_cap: str # 'capped' or 'uncapped'
    group: str    # Bullish/Bearish/Neutral/Other

@dataclass
class StratResult:
    key: str; name: str; style: str; group: str
    credit_debit_sh: float                  # +credit (sell) / -debit (buy)
    be_low: Optional[float]; be_high: Optional[float]
    max_profit_sh: Optional[float]; max_loss_sh: Optional[float]  # per share
    roi: Optional[float]
    strikes: Dict[str, int]                 # legs / strikes
    pop: Optional[float]                    # probability of profit (approx)
    notes: str                              # short note on rule used

# ------------------------------ NSE Fetch ------------------------------------

NSE_HEADERS = {
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
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
            resp = sess.get(url, headers={**NSE_HEADERS,"referer":"https://www.nseindia.com/option-chain"}, timeout=12)
            if resp.status_code == 200: break
            time.sleep(sleep_s); sess = _nse_session()
        if resp is None or resp.status_code != 200: return None, None, None
        data = resp.json()
        rec = data.get("records", {})
        underlying = rec.get("underlyingValue")
        if not underlying: return None, None, None
        underlying = float(underlying)

        today = dt.date.today(); target = today + dt.timedelta(days=horizon_days)
        exps = rec.get("expiryDates") or []
        def p(x):
            try: return dt.datetime.strptime(x, "%d-%b-%Y").date()
            except: return None
        exps = [(s, p(s)) for s in exps if p(s)]
        if not exps: return underlying, None, None
        chosen_exp, _ = min(exps, key=lambda x: abs((x[1]-target).days))

        chain = rec.get("data", [])
        rows = [r for r in chain if r.get("expiryDate")==chosen_exp]
        if not rows:
            rows = (data.get("filtered", {}) or {}).get("data", [])
            rows = [r for r in rows if r.get("expiryDate")==chosen_exp]
        if not rows: return underlying, None, chosen_exp

        atm_row, best = None, 1e18
        for r0 in rows:
            sp = r0.get("strikePrice"); 
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
    try:
        sess = _nse_session()
        resp = sess.get(LOT_CSV_URL, headers=NSE_HEADERS, timeout=12)
        if resp.status_code != 200:
            time.sleep(1.0); sess = _nse_session()
            resp = sess.get(LOT_CSV_URL, headers=NSE_HEADERS, timeout=12)
        if resp.status_code != 200: return {}
        rdr = csv.DictReader(io.StringIO(resp.text))
        out: Dict[str,int] = {}
        for row in rdr:
            sym = (row.get("SYMBOL") or row.get("Symbol") or row.get("Underlying") or "").strip().upper()
            lot = (row.get("MARKET LOT") or row.get("MarketLot") or row.get("LOT_SIZE") or
                   row.get("Lot Size") or row.get("Market Lot") or "")
            if not sym or not lot: continue
            try: out[sym] = int(float(lot))
            except: pass
        return out
    except Exception:
        return {}

# ------------------------------ Helpers --------------------------------------

def stride_and_em(spot: float, iv: float, days: int):
    t = max(1e-9, days/365.0)
    em = spot * iv * math.sqrt(t)  # 1σ move
    stride = max(1, suggest_stride(spot))
    return t, em, stride

def price_call(S,K,t,r,iv,q): return bs_price(S,K,t,r,iv,q,"call")
def price_put (S,K,t,r,iv,q): return bs_price(S,K,t,r,iv,q,"put")

def pop_between(S, em, lo, hi):
    # Approximate: ST ~ Normal(mean=S, sd=em)
    if lo is None and hi is None: return None
    def z(x): return (x - S) / (em if em>1e-9 else 1.0)
    lo_z = -10 if lo is None else z(lo)
    hi_z =  10 if hi is None else z(hi)
    return max(0.0, min(1.0, norm_cdf(hi_z) - norm_cdf(lo_z)))

# --------------------------- Strategy Generators -----------------------------

def width_from_pct(S, pct, stride):
    return max(stride*2, round_to_stride(pct*S, stride))

def mk_result(key,name,style,group,cd,beL,beH,maxP,maxL,roi,strikes,pop,notes):
    return StratResult(key=key,name=name,style=style,group=group,
                       credit_debit_sh=cd,be_low=beL,be_high=beH,
                       max_profit_sh=maxP,max_loss_sh=maxL,roi=roi,
                       strikes=strikes,pop=pop,notes=notes)

def gen_all_strats(S, iv, days, r, q,
                   sigma_ic, sigma_bps, sigma_ls,
                   w_ic_pct, w_bps_pct,
                   allow_naked: bool):
    t, em, stride = stride_and_em(S, iv, days)

    results: List[StratResult] = []

    # ---------- Neutral: Short Iron Condor ----------
    w_ic = width_from_pct(S, w_ic_pct, stride)
    sc = round_to_stride(S + sigma_ic*em, stride); lc = sc + w_ic
    sp = round_to_stride(S - sigma_ic*em, stride); lp = sp - w_ic
    scP, lcP = price_call(S,sc,t,r,iv,q), price_call(S,lc,t,r,iv,q)
    spP, lpP = price_put (S,sp,t,r,iv,q), price_put (S,lp,t,r,iv,q)
    credit = (scP-lcP)+(spP-lpP); width = max(lc-sc, sp-lp)
    risk = max(1e-9, width - credit); roi = credit/risk
    beL, beH = sp-credit, sc+credit
    results.append(mk_result("IC","Short Iron Condor","credit","Neutral",
                             credit,beL,beH,credit,risk,roi,
                             {"SP":sp,"LP":lp,"SC":sc,"LC":lc},
                             pop_between(S,em,beL,beH), f"±{sigma_ic:.2f}σ; width≈{w_ic_pct*100:.1f}%"))

    # ---------- Neutral: Short Iron Butterfly ----------
    atm = round_to_stride(S, stride)
    wing = w_ic  # reuse width
    sc, lc = atm, atm+wing
    sp, lp = atm, atm-wing
    scP, lcP = price_call(S,sc,t,r,iv,q), price_call(S,lc,t,r,iv,q)
    spP, lpP = price_put (S,sp,t,r,iv,q), price_put (S,lp,t,r,iv,q)
    credit = (scP-lcP)+(spP-lpP); width = wing
    risk = max(1e-9, width - credit); roi = credit/risk
    beL, beH = atm-credit, atm+credit
    results.append(mk_result("IB","Short Iron Butterfly","credit","Neutral",
                             credit,beL,beH,credit,risk,roi,
                             {"SP":sp,"LP":lp,"SC":sc,"LC":lc},
                             pop_between(S,em,beL,beH),"ATM short straddle + wings"))

    # ---------- Neutral: Short Strangle / Short Straddle (naked) ----------
    if allow_naked:
        ss_putK  = round_to_stride(S - 1.1*em, stride)
        ss_callK = round_to_stride(S + 1.1*em, stride)
        cr = price_put(S,ss_putK,t,r,iv,q) + price_call(S,ss_callK,t,r,iv,q)
        beL, beH = ss_putK - cr, ss_callK + cr
        results.append(mk_result("SS","Short Strangle","credit","Neutral",
                                 cr,beL,beH,None,None,None,
                                 {"SP":ss_putK,"SC":ss_callK},
                                 pop_between(S,em,beL,beH),"±1.1σ naked"))
        sstrdK = atm; cr2 = price_put(S,sstrdK,t,r,iv,q) + price_call(S,sstrdK,t,r,iv,q)
        beL2, beH2 = atm-cr2, atm+cr2
        results.append(mk_result("SSTRD","Short Straddle","credit","Neutral",
                                 cr2,beL2,beH2,None,None,None,
                                 {"K":sstrdK}, pop_between(S,em,beL2,beH2),"ATM naked"))

    # ---------- Long Straddle / Long Strangle ----------
    kC = round_to_stride(S + sigma_ls*em, stride)
    kP = round_to_stride(S - sigma_ls*em, stride)
    debit_ls = price_call(S,kC,t,r,iv,q)+price_put(S,kP,t,r,iv,q)
    results.append(mk_result("LS","Long Strangle","debit","Neutral",
                             -debit_ls,kP-debit_ls,kC+debit_ls,None,debit_ls,None,
                             {"KC":kC,"KP":kP}, 1 - pop_between(S,em,kP-debit_ls,kC+debit_ls),
                             f"±{sigma_ls:.2f}σ long-vol"))
    kA = atm
    debit_lstrd = price_call(S,kA,t,r,iv,q)+price_put(S,kA,t,r,iv,q)
    results.append(mk_result("LSTRD","Long Straddle","debit","Neutral",
                             -debit_lstrd,kA-debit_lstrd,kA+debit_lstrd,None,debit_lstrd,None,
                             {"K":kA}, 1 - pop_between(S,em,kA-debit_lstrd,kA+debit_lstrd),
                             "ATM long-vol"))

    # ---------- Bullish spreads ----------
    w_b = width_from_pct(S, w_bps_pct, stride)

    # Bull Put Spread (credit)
    sp2 = round_to_stride(S - 1.0*em, stride); lp2 = sp2 - w_b
    cr_bps = price_put(S,sp2,t,r,iv,q) - price_put(S,lp2,t,r,iv,q)
    risk = max(1e-9, (sp2-lp2) - cr_bps); roi = cr_bps/risk; be = sp2 - cr_bps
    results.append(mk_result("BPS","Bull Put Spread","credit","Bullish",
                             cr_bps,be,None,cr_bps,risk,roi,
                             {"SP":sp2,"LP":lp2}, pop_between(S,em,be,None),"OTM put spread"))

    # Bull Call Spread (debit)
    lc1 = round_to_stride(S + 0.2*em, stride); sc1 = lc1 + w_b
    debit_bcs = price_call(S,lc1,t,r,iv,q) - price_call(S,sc1,t,r,iv,q)
    maxP = (sc1 - lc1) - debit_bcs; be = lc1 + debit_bcs
    results.append(mk_result("BCS","Bull Call Spread","debit","Bullish",
                             -debit_bcs,be,None,maxP,debit_bcs,maxP/max(debit_bcs,1e-9),
                             {"LC":lc1,"SC":sc1}, 1 - pop_between(S,em,None,be),"OTM call spread"))

    # Call Ratio Backspread (buy 2 OTM calls, sell 1 ATM call) – long convexity
    k_sell = atm; k_buy = round_to_stride(S + 1.0*em, stride)
    debit = 2*price_call(S,k_buy,t,r,iv,q) - price_call(S,k_sell,t,r,iv,q)
    # breakeven high approximately: k_sell + |debit|
    beH = k_sell + max(0.0,debit)
    results.append(mk_result("CRB","Call Ratio Backspread","debit","Bullish",
                             -debit,None,beH,None,max(0.0,debit),None,
                             {"SellC":k_sell,"BuyC":k_buy,"Ratio":"1x2"},
                             None,"ATM short 1, buy 2× OTM"))

    # Long Synthetic Future (call - put @ ATM)
    syn_debit = price_call(S,atm,t,r,iv,q) - price_put(S,atm,t,r,iv,q)
    results.append(mk_result("LSYNF","Long Synthetic Future","other","Bullish",
                             syn_debit,None,None,None,None,None,
                             {"K":atm},None,"Long C, Short P (ATM)"))

    # ---------- Bearish spreads ----------
    # Bear Call Spread (credit)
    sc2 = round_to_stride(S + 1.0*em, stride); lc2 = sc2 + w_b
    cr_bcs2 = price_call(S,sc2,t,r,iv,q) - price_call(S,lc2,t,r,iv,q)
    risk2 = max(1e-9, (lc2-sc2) - cr_bcs2); roi2 = cr_bcs2/risk2; be2 = sc2 + cr_bcs2
    results.append(mk_result("BECS","Bear Call Spread","credit","Bearish",
                             cr_bcs2,None,be2,cr_bcs2,risk2,roi2,
                             {"SC":sc2,"LC":lc2}, 1 - pop_between(S,em,None,be2),"OTM call spread"))

    # Bear Put Spread (debit)
    sp3 = round_to_stride(S - 0.2*em, stride); lp3 = sp3 - w_b
    debit_beps = price_put(S,sp3,t,r,iv,q) - price_put(S,lp3,t,r,iv,q)
    maxP3 = (sp3 - lp3) - debit_beps; be3 = sp3 - debit_beps
    results.append(mk_result("BEPS","Bear Put Spread","debit","Bearish",
                             -debit_beps,be3,None,maxP3,debit_beps,maxP3/max(debit_beps,1e-9),
                             {"SP":sp3,"LP":lp3}, pop_between(S,em,None,be3),"OTM put spread"))

    # Put Ratio Backspread (buy 2 OTM puts, sell 1 ATM put)
    k_sellp = atm; k_buyp = round_to_stride(S - 1.0*em, stride)
    debitp = 2*price_put(S,k_buyp,t,r,iv,q) - price_put(S,k_sellp,t,r,iv,q)
    beL = k_sellp - max(0.0,debitp)
    results.append(mk_result("PRB","Put Ratio Backspread","debit","Bearish",
                             -debitp,beL,None,None,max(0.0,debitp),None,
                             {"SellP":k_sellp,"BuyP":k_buyp,"Ratio":"1x2"},None,"ATM short 1, buy 2× OTM"))

    # Short Synthetic Future (short call + long put @ ATM)
    syn_credit = price_call(S,atm,t,r,iv,q) - price_put(S,atm,t,r,iv,q)
    results.append(mk_result("SSYNF","Short Synthetic Future","other","Bearish",
                             -syn_credit,None,None,None,None,None,
                             {"K":atm},None,"Short C, Long P (ATM)"))

    # ---------- Naked directional (optional) ----------
    if allow_naked:
        cr_put  = price_put (S, round_to_stride(S-1.0*em,stride), t,r,iv,q)
        cr_call = price_call(S, round_to_stride(S+1.0*em,stride), t,r,iv,q)
        results.append(mk_result("SELL_PUT","Sell Put (naked)","credit","Bullish",
                                 cr_put, round_to_stride(S-1.0*em,stride)-cr_put, None, cr_put, None, None,
                                 {"SP":round_to_stride(S-1.0*em,stride)}, None,""))
        results.append(mk_result("SELL_CALL","Sell Call (naked)","credit","Bearish",
                                 cr_call, None, round_to_stride(S+1.0*em,stride)+cr_call, cr_call, None, None,
                                 {"SC":round_to_stride(S+1.0*em,stride)}, None,""))

    # ---------- Butterflies / Condors (long debit) ----------
    # Bull Butterfly (calls): K1 < K2 < K3
    k2 = round_to_stride(S + 0.5*em, stride); k1 = k2 - w_b; k3 = k2 + w_b
    debit_bfly = price_call(S,k1,t,r,iv,q) - 2*price_call(S,k2,t,r,iv,q) + price_call(S,k3,t,r,iv,q)
    beL = k1 + debit_bfly; beH = k3 - debit_bfly; maxP = w_b - debit_bfly
    results.append(mk_result("BULL_BFLY","Bull Butterfly (calls)","debit","Bullish",
                             -debit_bfly,beL,beH,maxP,debit_bfly,maxP/max(debit_bfly,1e-9),
                             {"K1":k1,"K2":k2,"K3":k3}, pop_between(S,em,beL,beH),"1-2-1"))

    # Bear Butterfly (puts)
    k2p = round_to_stride(S - 0.5*em, stride); k1p = k2p - w_b; k3p = k2p + w_b
    debit_bflyp = price_put(S,k1p,t,r,iv,q) - 2*price_put(S,k2p,t,r,iv,q) + price_put(S,k3p,t,r,iv,q)
    beL = k1p + debit_bflyp; beH = k3p - debit_bflyp; maxP = w_b - debit_bflyp
    results.append(mk_result("BEAR_BFLY","Bear Butterfly (puts)","debit","Bearish",
                             -debit_bflyp,beL,beH,maxP,debit_bflyp,maxP/max(debit_bflyp,1e-9),
                             {"K1":k1p,"K2":k2p,"K3":k3p}, pop_between(S,em,beL,beH),"1-2-1"))

    # Bull Condor (calls) – debit
    kL1 = round_to_stride(S + 0.2*em, stride); kS1 = kL1 + w_b
    kS2 = kS1 + w_b; kL2 = kS2 + w_b
    debit_cnd = (price_call(S,kL1,t,r,iv,q) - price_call(S,kS1,t,r,iv,q)) \
              + (price_call(S,kL2,t,r,iv,q) - price_call(S,kS2,t,r,iv,q))
    # approx: maxP occurs between short strikes: (kS2-kS1) - debit
    maxP = (kS2 - kS1) - debit_cnd
    results.append(mk_result("BULL_CONDOR","Bull Condor (calls)","debit","Bullish",
                             -debit_cnd,None,None,maxP,debit_cnd,None,
                             {"L1":kL1,"S1":kS1,"S2":kS2,"L2":kL2},None,"two call spreads"))

    # Bear Condor (puts) – debit
    pL1 = round_to_stride(S - 0.2*em, stride); pS1 = pL1 - w_b
    pS2 = pS1 - w_b; pL2 = pS2 - w_b
    debit_cndp = (price_put(S,pL1,t,r,iv,q) - price_put(S,pS1,t,r,iv,q)) \
               + (price_put(S,pL2,t,r,iv,q) - price_put(S,pS2,t,r,iv,q))
    maxPp = (pS1 - pS2) - debit_cndp
    results.append(mk_result("BEAR_CONDOR","Bear Condor (puts)","debit","Bearish",
                             -debit_cndp,None,None,maxPp,debit_cndp,None,
                             {"L1":pL1,"S1":pS1,"S2":pS2,"L2":pL2},None,"two put spreads"))

    # ---------- Strip / Strap ----------
    # Strip: long 2P + 1C at ATM (bearish vol)
    strip_debit = 2*price_put(S,atm,t,r,iv,q) + price_call(S,atm,t,r,iv,q)
    results.append(mk_result("STRIP","Strip (2P+1C @ATM)","debit","Other",
                             -strip_debit,None,None,None,strip_debit,None,
                             {"K":atm},None,"bearish long-vol tilt"))
    # Strap: long 2C + 1P at ATM (bullish vol)
    strap_debit = 2*price_call(S,atm,t,r,iv,q) + price_put(S,atm,t,r,iv,q)
    results.append(mk_result("STRAP","Strap (2C+1P @ATM)","debit","Other",
                             -strap_debit,None,None,None,strap_debit,None,
                             {"K":atm},None,"bullish long-vol tilt"))

    return results, em, stride

# ------------------------------ Ranking --------------------------------------

def score_strategy(res: StratResult, S: float, em: float) -> float:
    """Composite score: POP (0..1), ROI scaled, and BE distance / risk sanity."""
    pop = res.pop if res.pop is not None else 0.5
    roi = res.roi if res.roi is not None else 0.0
    # Breakeven distance normalized by σ
    be_span = 0.0
    if res.be_low is not None and res.be_high is not None:
        be_span = max(0.0, (res.be_high - res.be_low) / (2*em + 1e-9))
    elif res.be_low is not None:
        be_span = max(0.0, (S - res.be_low) / (2*em + 1e-9))
    elif res.be_high is not None:
        be_span = max(0.0, (res.be_high - S) / (2*em + 1e-9))

    # Cap unlimited risk strategies by giving small base weight (they’re off by default anyway)
    uncapped_penalty = 0.15 if (res.max_loss_sh is None and res.style=="credit") else 0.0

    # Weighted sum
    return 0.55*pop + 0.30*max(0.0, min(roi, 2.0)) + 0.20*min(be_span,1.0) - uncapped_penalty

# ------------------------------ HTML (Top picks) -----------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IV Agent – Top Strategies</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{--bg:#070b16;--card:#0f1629;--panel:#0b1222;--txt:#e8ecf6;--muted:#a4afc6;--line:#1b2a4a;--accent:#7aa2f7;
        --ic:#22d3ee;--bps:#a7f3d0;--ls:#f9a8d4;--c1:#60a5fa;--c2:#34d399;--c3:#fbbf24;--c4:#f472b6;--c5:#22d3ee;}
  *{box-sizing:border-box} body{margin:0;background:radial-gradient(1000px 600px at 10% -10%,#1c2550 0%,transparent 60%),
  radial-gradient(900px 600px at 110% 10%,#1d3a3f 0%,transparent 55%),var(--bg);color:var(--txt);
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Ubuntu,Arial,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:24px}
  h1{margin:0 0 4px 0;font-size:28px;letter-spacing:.3px}
  .meta{color:var(--muted);margin-bottom:18px}
  .toolbar{display:flex;gap:10px;align-items:center;margin:6px 0 20px 0}
  .search{flex:1;max-width:420px;background:#0b1427;border:1px solid #1d2a49;color:var(--txt);
          border-radius:12px;padding:10px 12px;outline:none}
  .btn{background:var(--accent);color:#fff;border:none;border-radius:12px;padding:10px 14px;cursor:pointer;text-decoration:none}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
  .card{background:linear-gradient(180deg,#111b33 0%,#0d1428 100%);border:1px solid #1a2747;border-radius:16px;padding:14px;
        cursor:pointer;transition:transform .12s ease,border-color .12s}
  .card:hover{transform:translateY(-2px);border-color:#2a3a6a}
  .sym{font-weight:700;font-size:18px;letter-spacing:.4px}
  .row{display:flex;justify-content:space-between;margin-top:8px;color:var(--muted)}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-top:10px;background:#173225;color:#8bffa9}
  .detail{position:fixed;inset:0;background:rgba(5,8,15,.65);display:none;align-items:flex-start;justify-content:center;
          padding:32px 16px;backdrop-filter:blur(4px)}
  .panel{width:min(1100px,96vw);max-height:92vh;overflow:auto;background:var(--panel);border:1px solid #1c294b;border-radius:18px;padding:20px}
  .panelhead{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:12px}
  .title{font-size:22px;font-weight:700}.sub{color:var(--muted)}.close{background:#25365e;border:1px solid #334876;color:#dbe4ff;border-radius:10px;padding:8px 10px;cursor:pointer}
  .cols{display:grid;grid-template-columns:1.3fr .7fr;gap:16px}@media (max-width:900px){.cols{grid-template-columns:1fr}}
  .box{background:#0e1933;border:1px solid #1c2a51;border-radius:14px;padding:12px}
  canvas{width:100%;height:360px;background:#0b1427;border:1px solid #1b2a4a;border-radius:14px}
  .legend{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}
  .chip{display:flex;align-items:center;gap:6px;background:#162243;border:1px solid #233463;border-radius:999px;padding:6px 10px;cursor:pointer}
  .chip.active{outline:2px solid #2f4aa8}.dot{width:10px;height:10px;border-radius:50%}
  .tab{display:inline-block;margin:0 8px 8px 0;padding:6px 10px;border-radius:999px;background:#14203f;border:1px solid #233463;cursor:pointer}
  .tab.active{outline:2px solid #2f4aa8}
  table{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px}
  th,td{padding:8px;border-bottom:1px solid #1e2846;text-align:left}
</style>
</head>
<body>
<div class="wrap">
  <h1>IV Agent – Top Strategies</h1>
  <div class="meta">Generated: __GENERATED__ · Horizon: __DAYS__ days · Lots=1 · Data: NSE · <a class="btn" href="__CSV_NAME__" download>CSV</a></div>
  <div class="toolbar"><input id="q" class="search" placeholder="Search symbol…"/></div>
  <div id="cards" class="grid"></div>
</div>

<div id="detail" class="detail">
  <div class="panel">
    <div class="panelhead">
      <div><div class="title" id="d_sym">—</div><div class="sub" id="d_meta">—</div></div>
      <button class="close" onclick="closeDetail()">Close</button>
    </div>
    <div class="cols">
      <div class="box">
        <div id="tabs"></div>
        <canvas id="chart" width="900" height="360"></canvas>
        <div class="legend" id="legend"></div>
      </div>
      <div class="box">
        <div style="margin-bottom:6px;color:#cbd5e1"><b>Selected strategies</b> (per share / per lot)</div>
        <table id="tbl"><thead>
          <tr><th>Strategy</th><th>Credit/Debit</th><th>B/E</th><th>Max P/L</th><th>ROI%</th><th>POP%</th></tr>
        </thead><tbody></tbody></table>
        <div style="margin-top:10px"><span id="toggleAll" class="tab">Show all strategies</span></div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const TOPN = __TOPN__;

const cards = document.getElementById('cards');
const q = document.getElementById('q');
q.addEventListener('input', e=>{
  const t=e.target.value.toLowerCase();
  Array.from(cards.children).forEach(c=>{c.style.display=c.textContent.toLowerCase().includes(t)?'':''})
});

function pill(s){const d=document.createElement('div');d.className='pill';d.textContent=s;return d;}
DATA.forEach((d,i)=>{
  const card=document.createElement('div');card.className='card';card.onclick=()=>openDetail(i);
  card.innerHTML=`<div class="sym">${d.symbol}</div>
  <div class="row"><span>Spot</span><span>₹${d.spot.toFixed(2)}</span></div>
  <div class="row"><span>IV</span><span>${(d.iv*100).toFixed(1)}%</span></div>`;
  const p=pill(d.top.map(x=>x.name).join(' · ')); card.appendChild(p); cards.appendChild(card);
});

const detail=document.getElementById('detail');
const d_sym=document.getElementById('d_sym'); const d_meta=document.getElementById('d_meta');
const tbl=document.querySelector('#tbl tbody'); const legend=document.getElementById('legend'); const tabs=document.getElementById('tabs');
const toggleAll=document.getElementById('toggleAll');
let CUR=null, SHOW_ALL=false, sel=[];

function openDetail(i){
  CUR=DATA[i]; SHOW_ALL=false;
  d_sym.textContent=CUR.symbol;
  d_meta.textContent=`Expiry: ${CUR.expiry||'—'} · Lot: ${CUR.lot} · Spot: ₹${CUR.spot.toFixed(2)} · IV ${(CUR.iv*100).toFixed(1)}%`;
  sel=CUR.top.map(x=>x.key);
  renderTabs(); renderLegend(); renderTable(); draw();
  detail.style.display='flex';
}
function closeDetail(){ detail.style.display='none'; }

function renderTabs(){
  tabs.innerHTML='';
  ['±1σ','±1.5σ','±2σ'].forEach((t,idx)=>{
    const span=document.createElement('span');span.className='tab'+(idx==0?' active':'');span.textContent=t;span.dataset.m=[1,1.5,2][idx];
    span.onclick=(e)=>{Array.from(tabs.children).forEach(c=>c.classList.remove('active')); e.target.classList.add('active'); draw(parseFloat(e.target.dataset.m));}
    tabs.appendChild(span);
  });
}
function colorFor(i){return ['#60a5fa','#34d399','#fbbf24','#f472b6','#22d3ee','#e5e7eb'][i%6];}
function renderLegend(){
  legend.innerHTML='';
  const arr = SHOW_ALL ? CUR.all : CUR.top;
  arr.forEach((s,i)=>{
    const chip=document.createElement('span');chip.className='chip'+(sel.includes(s.key)?' active':'');
    chip.innerHTML=`<span class="dot" style="background:${colorFor(i)}"></span>${s.name}`;
    chip.onclick=()=>{ if(sel.includes(s.key)) sel=sel.filter(k=>k!==s.key); else sel.push(s.key); draw(); renderTable(); chip.classList.toggle('active'); };
    legend.appendChild(chip);
  });
  toggleAll.textContent = SHOW_ALL ? 'Show only top' : 'Show all strategies';
  toggleAll.onclick=()=>{ SHOW_ALL=!SHOW_ALL; renderLegend(); renderTable(); draw(); };
}

function fmt(x){return x==null?'—':(typeof x==='number'?x.toFixed(2):x);}
function renderTable(){
  tbl.innerHTML=''; const arr = SHOW_ALL ? CUR.all : CUR.top;
  arr.filter(s=>sel.includes(s.key)).forEach(s=>{
    const tr=document.createElement('tr');
    const cd = (s.credit_debit_sh>=0?'+':'')+s.credit_debit_sh.toFixed(2)+` / `+((s.credit_debit_sh)*CUR.lot).toFixed(2);
    const be = (s.be_low? s.be_low.toFixed(2):'—')+' / '+(s.be_high? s.be_high.toFixed(2):'—');
    const mpl = (s.max_profit_sh is null? 'Uncapped' : s.max_profit_sh.toFixed(2)) + ' / ' + (s.max_profit_sh is null? 'Uncapped' : (s.max_profit_sh*CUR.lot).toFixed(2));
  });
  arr.filter(s=>sel.includes(s.key)).forEach(s=>{
    const tr=document.createElement('tr');
    const cd = (s.credit_debit_sh>=0?'+':'')+s.credit_debit_sh.toFixed(2)+' / '+((s.credit_debit_sh)*CUR.lot).toFixed(2);
    const be = (s.be_low!=null? s.be_low.toFixed(2):'—')+' / '+(s.be_high!=null? s.be_high.toFixed(2):'—');
    const maxp = (s.max_profit_sh==null? 'Uncapped' : s.max_profit_sh.toFixed(2))+' / '+(s.max_profit_sh==null? 'Uncapped' : (s.max_profit_sh*CUR.lot).toFixed(2));
    const maxl = (s.max_loss_sh==null? '—' : s.max_loss_sh.toFixed(2))+' / '+(s.max_loss_sh==null? '—' : (s.max_loss_sh*CUR.lot).toFixed(2));
    const roi = s.roi==null? '—' : (s.roi*100).toFixed(1);
    const pop = s.pop==null? '—' : (s.pop*100).toFixed(1);
    tr.innerHTML=`<td>${s.name}</td><td>${cd}</td><td>${be}</td><td>${maxp} · ${maxl}</td><td>${roi}</td><td>${pop}</td>`;
    tbl.appendChild(tr);
  });
}

function payoffLine(s, ST){
  const L=CUR.lot;
  const K=s.strikes||{};
  function max(a,b){return a>b?a:b;} function min(a,b){return a<b?a:b;}
  function callPay(k){return max(0, ST-k);} function putPay(k){return max(0, k-ST);}
  // Map by key
  switch(s.key){
    case 'IC': {
      const sp=K.SP, lp=K.LP, sc=K.SC, lc=K.LC; const credit=s.credit_debit_sh;
      let callLoss=0; if (ST>sc && ST<lc) callLoss=ST-sc; else if (ST>=lc) callLoss=(lc-sc);
      let putLoss=0;  if (ST<sp && ST>lp) putLoss=sp-ST; else if (ST<=lp) putLoss=(sp-lp);
      return (credit - callLoss - putLoss)*L;
    }
    case 'IB': {
      const sp=K.SP, lp=K.LP, sc=K.SC, lc=K.LC; const credit=s.credit_debit_sh;
      let callLoss=0; if (ST>sc && ST<lc) callLoss=ST-sc; else if (ST>=lc) callLoss=(lc-sc);
      let putLoss=0;  if (ST<sp && ST>lp) putLoss=sp-ST; else if (ST<=lp) putLoss=(sp-lp);
      return (credit - callLoss - putLoss)*L;
    }
    case 'BPS': {
      const sp=K.SP, lp=K.LP; const cr=s.credit_debit_sh;
      let loss=0; if (ST<sp && ST>lp) loss=sp-ST; else if (ST<=lp) loss=(sp-lp);
      return (cr - loss)*L;
    }
    case 'BCS': {
      const lc=K.LC, sc=K.SC; const deb=-s.credit_debit_sh;
      const val = min(max(0, ST-lc), sc-lc); return (val - deb)*L;
    }
    case 'BECS': {
      const sc=K.SC, lc=K.LC; const cr=s.credit_debit_sh;
      let loss=0; if (ST>sc && ST<lc) loss=ST-sc; else if (ST>=lc) loss=(lc-sc);
      return (cr - loss)*L;
    }
    case 'BEPS': {
      const sp=K.SP, lp=K.LP; const deb=-s.credit_debit_sh;
      const val = min(max(0, sp-ST), sp-lp); return (val - deb)*L;
    }
    case 'LSTRD': {
      const K0=K.K; const deb=-s.credit_debit_sh;
      return (callPay(K0)+putPay(K0) - deb)*L;
    }
    case 'LS': {
      const kc=K.KC, kp=K.KP; const deb=-s.credit_debit_sh;
      return (callPay(kc)+putPay(kp) - deb)*L;
    }
    case 'SSTRD': {
      const K0=K.K; const cr=s.credit_debit_sh;
      return (cr - callPay(K0) - putPay(K0))*L;
    }
    case 'SS': {
      const sp=K.SP, sc=K.SC; const cr=s.credit_debit_sh;
      return (cr - max(0,sp-ST) - max(0,ST-sc))*L;
    }
    case 'BULL_BFLY': {
      const k1=K.K1,k2=K.K2,k3=K.K3; const deb=-s.credit_debit_sh;
      const val = min(max(0,ST-k1),k2-k1) + min(max(0,ST-k2),k3-k2) - min(max(0,ST-k2),k3-k2);
      // simpler: two call spreads centered; approximate peak at k2
      const wing=k2-k1; const peak = max(0, wing - deb);
      const tri = Math.max(0, wing - Math.abs(ST-k2))  # approx triangle
    }
  }
  // Fallback simple triangle for butterflies/condors if not matched
  return 0;
}

function draw(mult=1.0){
  const c=document.getElementById('chart'); const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  const arr = (SHOW_ALL?CUR.all:CUR.top).filter(s=>sel.includes(s.key));
  if(!arr.length){return;}
  const S=CUR.spot, em=CUR.em, L=CUR.lot; const range=mult*em; const xMin=Math.max(0,S-range*1.1), xMax=S+range*1.1;
  const N=360, xs=[], series=arr.map(()=>[]);
  for(let i=0;i<=N;i++){
    const ST = xMin + (xMax-xMin)*i/N; xs.push(ST);
    arr.forEach((s,idx)=>{ series[idx].push(payoffLine(s,ST)); });
  }
  let yMin=0,yMax=0; series.forEach(a=>a.forEach(v=>{yMin=Math.min(yMin,v); yMax=Math.max(yMax,v)}));
  if(yMin===yMax){yMin-=1;yMax+=1} const pad=(yMax-yMin)*0.12; yMin-=pad; yMax+=pad;
  const X=x=>((x-xMin)/(xMax-xMin))*(c.width-60)+40; const Y=y=>c.height-30 - ((y-yMin)/(yMax-yMin))*(c.height-60);
  ctx.strokeStyle="#1b2a4a"; ctx.strokeRect(40,20,c.width-80,c.height-50);
  const xS=X(S); ctx.beginPath(); ctx.moveTo(xS,20); ctx.lineTo(xS,c.height-30); ctx.stroke(); const y0=Y(0); ctx.beginPath(); ctx.moveTo(40,y0); ctx.lineTo(c.width-40,y0); ctx.stroke();
  ctx.fillStyle="#a4afc6"; ctx.font="12px Segoe UI, Roboto"; ctx.fillText(`S=₹${S.toFixed(2)}`, xS+4, 30);
  arr.forEach((s,idx)=>{
    const col=['#60a5fa','#34d399','#fbbf24','#f472b6','#22d3ee','#e5e7eb'][idx%6];
    ctx.beginPath(); ctx.strokeStyle=col; ctx.lineWidth=2;
    series[idx].forEach((v,i)=>{ const px=X(xs[i]), py=Y(v); if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py); });
    ctx.stroke();
  });
}

</script>
</body>
</html>
"""

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

def write_csv(out_path: str, rows: List[Dict[str, str]]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

# ------------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="IV Agent – Multi-strategy (NSE) with website of top picks")
    ap.add_argument("--tickers", required=True, type=str)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--risk_free", type=float, default=0.07)
    ap.add_argument("--yield_div", type=float, default=0.00)
    ap.add_argument("--sigma_ic", type=float, default=1.2)
    ap.add_argument("--sigma_bps", type=float, default=1.0)
    ap.add_argument("--sigma_ls", type=float, default=1.0)
    ap.add_argument("--width_pct_ic", type=float, default=0.02)
    ap.add_argument("--width_pct_bps", type=float, default=0.02)
    ap.add_argument("--default_iv", type=float, default=0.25)
    ap.add_argument("--iv_csv", type=str, default="")
    ap.add_argument("--spot_csv", type=str, default="")
    ap.add_argument("--live_iv", action="store_true")
    ap.add_argument("--use_yfinance_fallback", action="store_true")
    ap.add_argument("--allow_naked", action="store_true")
    ap.add_argument("--top_n", type=int, default=3)
    ap.add_argument("--out_csv", type=str, default="iv_agent_output.csv")
    ap.add_argument("--html_out", type=str, default="iv_agent_output.html")
    ap.add_argument("--open_html", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    lot_map = fetch_lot_sizes()

    spot_map = read_csv_map(args.spot_csv, "ticker", "spot") if args.spot_csv else {}
    iv_map   = read_csv_map(args.iv_csv,   "ticker", "iv")   if args.iv_csv   else {}

    csv_rows: List[Dict[str,str]] = []
    js_rows: List[dict] = []

    for sym in tickers:
        key = normalize_symbol_key(sym)
        lot = lot_map.get(key, 1)

        # spot + iv
        spot = spot_map.get(sym); iv = iv_map.get(sym); chosen_exp = ""
        if args.live_iv:
            s_live, iv_live, exp_live = fetch_iv_spot_from_nse(sym, args.days)
            if s_live is not None: spot = s_live
            if iv_live is not None: iv = iv_live
            if exp_live: chosen_exp = exp_live
        if spot is None and args.use_yfinance_fallback:
            try:
                import yfinance as yf
                s = sym if sym.endswith(".NS") else f"{sym}.NS"
                data = yf.Ticker(s).history(period="1d")
                if data is not None and not data.empty:
                    spot = float(data["Close"].iloc[-1])
            except Exception: pass
        if iv is None: iv = args.default_iv
        if iv > 1.0: iv = iv/100.0
        if spot is None: continue

        # generate all & rank
        all_strats, em, stride = gen_all_strats(
            spot, iv, args.days, args.risk_free, args.yield_div,
            args.sigma_ic, args.sigma_bps, args.sigma_ls,
            args.width_pct_ic, args.width_pct_bps,
            args.allow_naked
        )
        # score
        scored = [(score_strategy(s, spot, em), s) for s in all_strats]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _,s in scored[:max(1,args.top_n)]]

        # CSV (top only – primary = top[0])
        primary = top[0]
        def fmt(x): return "" if x is None else f"{x:.2f}"
        csv_rows.append({
            "Symbol": sym, "Expiry(closest)": chosen_exp, "Lot": lot,
            "Spot": f"{spot:.2f}", "ATM IV": f"{iv:.4f}",
            "1σ Move(₹)": f"{em:.2f}", "Range ±1σ": f"{spot-em:.2f}–{spot+em:.2f}",
            "Top1 Strategy": primary.name, "Top1 POP%": "" if primary.pop is None else f"{primary.pop*100:.1f}",
            "Top1 ROI%": "" if primary.roi is None else f"{primary.roi*100:.1f}",
            "Top1 BE": f"{fmt(primary.be_low)} / {fmt(primary.be_high)}",
            "Top1 Credit/Debit (₹/sh)": f"{primary.credit_debit_sh:.2f}",
            "Top1 Max Profit (₹/sh)": "Uncapped" if primary.max_profit_sh is None else f"{primary.max_profit_sh:.2f}",
            "Top1 Max Loss (₹/sh)": "" if primary.max_loss_sh is None else f"{primary.max_loss_sh:.2f}",
        })

        # Website data
        def to_js(s: StratResult):
            return {
                "key": s.key, "name": s.name, "group": s.group, "style": s.style,
                "credit_debit_sh": float(f"{s.credit_debit_sh:.6f}"),
                "be_low": None if s.be_low is None else float(f"{s.be_low:.6f}"),
                "be_high": None if s.be_high is None else float(f"{s.be_high:.6f}"),
                "max_profit_sh": None if s.max_profit_sh is None else float(f"{s.max_profit_sh:.6f}"),
                "max_loss_sh": None if s.max_loss_sh is None else float(f"{s.max_loss_sh:.6f}"),
                "roi": None if s.roi is None else float(f"{s.roi:.6f}"),
                "pop": None if s.pop is None else float(f"{s.pop:.6f}"),
                "strikes": s.strikes, "notes": s.notes
            }
        js_rows.append({
            "symbol": sym, "expiry": chosen_exp, "spot": float(f"{spot:.6f}"),
            "iv": float(f"{iv:.6f}"), "lot": int(lot), "days": args.days, "em": float(f"{em:.6f}"),
            "top": [to_js(s) for s in top],
            "all": [to_js(s) for _,s in scored]
        })

    if not js_rows:
        print("No data generated. Check connectivity / symbols.")
        sys.exit(2)

    # Outputs
    write_csv(args.out_csv, csv_rows)
    html = (HTML.replace("__GENERATED__", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                 .replace("__DAYS__", str(args.days))
                 .replace("__CSV_NAME__", os.path.basename(args.out_csv))
                 .replace("__TOPN__", str(max(1,args.top_n)))
                 .replace("__DATA_JSON__", json.dumps(js_rows, ensure_ascii=False)))
    with open(args.html_out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved CSV : {os.path.abspath(args.out_csv)}")
    print(f"Saved HTML: {os.path.abspath(args.html_out)}")
    if args.open_html: webbrowser.open("file://" + os.path.abspath(args.html_out))

if __name__ == "__main__":
    main()
