#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IV Agent – NSE live IV → Full strategy set + ranking + website (Top Picks + Payoff + Greeks)

Included strategies (single-expiry unless noted):
  Neutral  : Short Iron Condor (IC), Short Iron Butterfly (IB), Short Strangle (SS*), Short Straddle (SSTRD*),
             Long Strangle (LS), Long Straddle (LSTRD), Batman (double fly), Double Plateau (double condor)
  Bullish  : Bull Put Spread (BPS), Bull Call Spread (BCS), Call Ratio Backspread (CRB), Bull Butterfly (calls),
             Bull Condor (calls), Long Synthetic Future (LSYNF), Jade Lizard (JL), Strap (2C+1P)
  Bearish  : Bear Call Spread (BECS), Bear Put Spread (BEPS), Put Ratio Backspread (PRB), Bear Butterfly (puts),
             Bear Condor (puts), Short Synthetic Future (SSYNF), Strip (2P+1C)
  Others   : Risk Reversal / Range Forward (RR), Long Iron Condor (LIC), Long Iron Butterfly (LIB)
  Calendars: Long Calendar (Calls) LCC, Long Calendar (Puts) LCP  (evaluated at NEAR expiry)

(* Naked shorts OFF by default. Enable with --allow_naked)

Homepage shows stock cards only; click a card → detail panel with:
  • Best strategy (first) + list of all other strategies
  • Payoff chart (per LOT) computed from legs (supports calendars)
  • Orders (BUY/SELL, NEAR/FAR, CALL/PUT, strike, qty)
  • Summary row (Credit/Debit, B/E, Max P/L per lot, ROI, POP)
  • P&L key-points table (Spot, B/Es, ±1σ, ±2σ)
  • Aggregate Greeks per lot (Δ, Γ, Θ/day, Vega)
  • Download P&L / Greeks of selected strategy as CSV

New in this version:
  • --strike_mode sigma|delta|prob|ev
    - sigma: original ±k·σ / width% approach (default)
    - delta: target |Δ| for short & wing strikes (e.g., 0.18 / 0.05)
    - prob : target tail probabilities under lognormal for shorts/wings
    - ev   : coarse-grid EV optimizer for the Iron Condor (keeps others sigma unless adapted below)
  • Robust breakeven finder + denser scan for POP
  • Optional delta/prob strike picking for BPS/BCS/BECS/BEPS

Run (example):
  python iv_agent.py --tickers "RELIANCE,ICICIBANK,SBIN" \
    --days 30 --live_iv --use_yfinance_fallback --top_n 3 --open_html \
    --strike_mode delta --short_delta 0.18 --wing_delta 0.05
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

def norm_pdf(x: float) -> float:
    return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

def bs_price(S: float, K: float, t: float, r: float, sigma: float, q: float = 0.0, typ: str = "call") -> float:
    S = max(1e-9, float(S)); K = max(1e-9, float(K)); t = max(1e-9, float(t)); sigma = max(1e-9, float(sigma))
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))
    d2 = d1 - sigma*math.sqrt(t)
    if typ == "call":
        return S*math.exp(-q*t)*norm_cdf(d1) - K*math.exp(-r*t)*norm_cdf(d2)
    else:
        return K*math.exp(-r*t)*norm_cdf(-d2) - S*math.exp(-q*t)*norm_cdf(-d1)

# ---- Extra Math Helpers (Delta / Tail-Prob / Inverse-Normal etc.) -----------

def inv_norm_cdf(p: float) -> float:
    """Approximate inverse CDF (Acklam). p in (0,1)."""
    p = min(max(p, 1e-12), 1-1e-12)
    a1=-3.969683028665376e+01; a2= 2.209460984245205e+02; a3=-2.759285104469687e+02
    a4= 1.383577518672690e+02; a5=-3.066479806614716e+01; a6= 2.506628277459239e+00
    b1=-5.447609879822406e+01; b2= 1.615858368580409e+02; b3=-1.556989798598866e+02
    b4= 6.680131188771972e+01; b5=-1.328068155288572e+01
    c1=-7.784894002430293e-03; c2=-3.223964580411365e-01; c3=-2.400758277161838e+00
    c4=-2.549732539343734e+00; c5= 4.374664141464968e+00; c6= 2.938163982698783e+00
    d1= 7.784695709041462e-03; d2= 3.224671290700398e-01; d3= 2.445134137142996e+00
    d4= 3.754408661907416e+00
    plow=0.02425; phigh=1-plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6)/((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q = p-0.5; r=q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q/(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)

def d1_val(S,K,t,r,sigma,q):
    return (math.log(S/K) + (r - q + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t))

def call_delta(S,K,t,r,sigma,q):
    return math.exp(-q*t)*norm_cdf(d1_val(S,K,t,r,sigma,q))

def put_delta(S,K,t,r,sigma,q):
    return math.exp(-q*t)*(norm_cdf(d1_val(S,K,t,r,sigma,q))-1.0)

def strike_for_target_call_delta(S,t,r,q,sigma,target_delta):
    """Find K s.t. call delta ~= target_delta (0..1)."""
    lo, hi = S*0.2, S*5.0
    for _ in range(60):
        mid = (lo+hi)/2.0
        d = call_delta(S, mid, t, r, sigma, q)
        if d > target_delta: lo = mid
        else: hi = mid
    return (lo+hi)/2.0

def strike_for_target_put_delta(S,t,r,q,sigma,target_abs_delta):
    """Find K for OTM put (delta negative; abs≈target_abs_delta)."""
    lo, hi = S*0.2, S*5.0
    for _ in range(60):
        mid = (lo+hi)/2.0
        d = abs(put_delta(S, mid, t, r, sigma, q))
        if d > target_abs_delta: hi = mid   # deeper ITM (bigger |Δ|)
        else: lo = mid
    return (lo+hi)/2.0

def strike_for_otm_prob(S,t,r,q,sigma, upper_tail_prob: float, is_call: bool) -> float:
    """Choose K so that P(ST > K)=p (call) or P(ST < K)=p (put) under lognormal."""
    mu = math.log(S) + (r - q - 0.5*sigma*sigma)*t
    s  = sigma*math.sqrt(t)
    if is_call:
        z = inv_norm_cdf(1.0 - upper_tail_prob)
        lnK = mu + s*z
    else:
        z = inv_norm_cdf(upper_tail_prob)
        lnK = mu + s*z
    return math.exp(lnK)

# --------------------------- Stride helpers ----------------------------------

def suggest_stride(spot: float) -> int:
    if spot < 200: return 1
    if spot < 500: return 5
    if spot < 1000: return 10
    if spot < 2000: return 20
    if spot < 4000: return 25
    return 50

def round_to_stride(x: float, stride: int) -> int:
    return int(round(x/stride)*stride)

# ---------------------------- Config / Aliases -------------------------------

NSE_SYMBOL_ALIASES = { "BAJAJ_AUTO": "BAJAJ-AUTO" }
def normalize_symbol_for_api(sym: str) -> str:
    s = sym.strip().upper(); s = NSE_SYMBOL_ALIASES.get(s, s); return quote(s, safe="")
def normalize_symbol_key(sym: str) -> str:
    s = sym.strip().upper(); return NSE_SYMBOL_ALIASES.get(s, s)

# ------------------------------ Data Models ----------------------------------

@dataclass
class Leg:
    kind: str        # 'call' or 'put'
    side: str        # 'long' or 'short'
    K: int
    qty: int         # 1,2,...
    t0: float        # years to its own expiry at entry
    iv0: float       # iv used at entry for pricing
    t_eval: float    # years remaining at chart evaluation (0 for single-expiry; >0 for calendars)
    iv_eval: float   # iv used at evaluation (far leg in calendars)

@dataclass
class Strat:
    key: str
    name: str
    group: str
    style: str       # 'credit' or 'debit' or 'other'
    legs: List[Leg]
    prem0_sh: float  # Σ(sign * price0)
    be_low: Optional[float]
    be_high: Optional[float]
    max_profit_sh: Optional[float]
    max_loss_sh: Optional[float]
    roi: Optional[float]
    pop: Optional[float]
    notes: str

# ------------------------------ NSE Fetch ------------------------------------

HDRS = {
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "accept-language": "en-US,en;q=0.9",
}

def _nse_session() -> requests.Session:
    s = requests.Session()
    s.get("https://www.nseindia.com/option-chain", headers=HDRS, timeout=10)
    return s

def fetch_chain_core(symbol: str) -> dict:
    sess = _nse_session()
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={normalize_symbol_for_api(symbol)}"
    for _ in range(3):
        r = sess.get(url, headers={**HDRS, "referer": "https://www.nseindia.com/option-chain"}, timeout=12)
        if r.status_code == 200: return r.json()
        time.sleep(1.2); sess = _nse_session()
    return {}

def parse_expiries(data: dict) -> List[Tuple[str, dt.date]]:
    out=[]; exps=(data.get("records",{}) or {}).get("expiryDates") or []
    for s in exps:
        try: out.append((s, dt.datetime.strptime(s, "%d-%b-%Y").date()))
        except: pass
    return out

def closest_two(exp_pairs: List[Tuple[str, dt.date]], target_date: dt.date) -> Tuple[Optional[str], Optional[str]]:
    if not exp_pairs: return None, None
    pairs=sorted(exp_pairs, key=lambda x: x[1])
    near=None; nxt=None
    near=min(pairs, key=lambda x: abs((x[1]-target_date).days))
    after=[p for p in pairs if p[1]>near[1]]
    nxt = after[0] if after else None
    return (near[0] if near else None, nxt[0] if nxt else None)

def atm_iv_for_exp(data: dict, exp: str, spot: float) -> Optional[float]:
    rows=[r for r in (data.get("records",{}).get("data") or []) if r.get("expiryDate")==exp]
    if not rows:
        rows=[r for r in (data.get("filtered",{}).get("data") or []) if r.get("expiryDate")==exp]
    if not rows: return None
    best=None; atm=None
    for r in rows:
        sp=r.get("strikePrice"); 
        if sp is None: continue
        d=abs(float(sp)-spot)
        if best is None or d<best:
            best=d; atm=r
    if not atm: return None
    ce=atm.get("CE",{}).get("impliedVolatility"); pe=atm.get("PE",{}).get("impliedVolatility")
    if ce is not None and pe is not None: return (float(ce)+float(pe))/200.0
    if ce is not None: return float(ce)/100.0
    if pe is not None: return float(pe)/100.0
    return None

def fetch_live(symbol: str, horizon_days: int) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[float], Optional[float]]:
    data = fetch_chain_core(symbol)
    spot = (data.get("records",{}) or {}).get("underlyingValue")
    if not spot: return None, None, None, None, None
    spot=float(spot)
    today=dt.date.today(); target=today+dt.timedelta(days=horizon_days)
    pairs=parse_expiries(data)
    near, nxt = closest_two(pairs, target)
    iv_near = atm_iv_for_exp(data, near, spot) if near else None
    iv_next = atm_iv_for_exp(data, nxt,  spot) if nxt  else None
    return spot, near, nxt, iv_near, iv_next

# ---------- Permitted lot size (official NSE CSV) ----------------------------

LOT_CSV_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
def fetch_lot_sizes() -> Dict[str, int]:
    try:
        sess=_nse_session()
        r=sess.get(LOT_CSV_URL, headers=HDRS, timeout=12)
        if r.status_code!=200:
            time.sleep(1.0); sess=_nse_session(); r=sess.get(LOT_CSV_URL, headers=HDRS, timeout=12)
        if r.status_code!=200: return {}
        rdr=csv.DictReader(io.StringIO(r.text)); out={}
        for row in rdr:
            sym=(row.get("SYMBOL") or row.get("Symbol") or row.get("Underlying") or "").strip().upper()
            lot=(row.get("MARKET LOT") or row.get("MarketLot") or row.get("LOT_SIZE") or row.get("Lot Size") or "")
            if sym and lot:
                try: out[sym]=int(float(lot))
                except: pass
        return out
    except Exception:
        return {}

# ------------------------------ Engine (legs) --------------------------------

def stride_and_em(spot: float, iv: float, days: int):
    t = max(1e-9, days/365.0)
    em = spot * iv * math.sqrt(t)  # 1σ move
    stride = max(1, suggest_stride(spot))
    return t, em, stride

def price_leg_now(S: float, r: float, q: float, leg: Leg, t_eval_override: Optional[float]=None, iv_override: Optional[float]=None) -> float:
    """Value of a single leg at evaluation time (t_eval years remaining). If t_eval=0 → intrinsic."""
    t_eval = leg.t_eval if t_eval_override is None else t_eval_override
    iv_use = leg.iv_eval if iv_override is None else iv_override
    sign = 1 if leg.side=="long" else -1
    if t_eval <= 1e-9:
        if leg.kind=="call": val = max(0.0, S - leg.K)
        else:                val = max(0.0, leg.K - S)
    else:
        val = bs_price(S, leg.K, t_eval, r, iv_use, q, "call" if leg.kind=="call" else "put")
    return sign * leg.qty * val

def price_leg_entry(S: float, r: float, q: float, leg: Leg) -> float:
    """Value at entry (to compute net premium)."""
    val = bs_price(S, leg.K, leg.t0, r, leg.iv0, q, "call" if leg.kind=="call" else "put")
    sign = 1 if leg.side=="long" else -1
    return sign * leg.qty * val

def pnl_at(S: float, r: float, q: float, legs: List[Leg], prem0: float) -> float:
    """Per-share P&L at evaluation point for spot= S."""
    value_now = sum(price_leg_now(S, r, q, L) for L in legs)
    return value_now - prem0

# ------- Robust breakevens + POP scan (denser grid for accuracy) --------------

def find_breakevens(xs: List[float], ys: List[float], S: float) -> Tuple[Optional[float], Optional[float]]:
    beL = None; beH = None
    roots = []
    for i in range(1, len(xs)):
        if ys[i-1] == 0.0:
            roots.append(xs[i-1]); continue
        if ys[i] == 0.0:
            roots.append(xs[i]); continue
        if ys[i-1]*ys[i] < 0:
            x1,x2=xs[i-1],xs[i]; y1,y2=ys[i-1],ys[i]
            xz = x1 - y1*(x2-x1)/(y2-y1)
            roots.append(xz)
    if roots:
        roots.sort()
        for r_ in roots:
            if r_ < S: beL = r_
            if r_ >= S and beH is None: beH = r_
    return beL, beH

def scan_metrics(S: float, em: float, r: float, q: float, legs: List[Leg], prem0: float) -> Tuple[float,float,Optional[float],Optional[float],float]:
    lo = max(0.0, S - 3.5*em); hi = S + 3.5*em
    N = 1400
    xs = [lo + (hi-lo)*i/N for i in range(N+1)]
    ys = [pnl_at(x, r, q, legs, prem0) for x in xs]
    maxP = max(ys); minP = min(ys)
    beL, beH = find_breakevens(xs, ys, S)
    # POP integrate
    def z(x): return (x - S) / (em if em>1e-9 else 1.0)
    pop_num = 0.0; pop_den = 0.0
    for i in range(N):
        x1, x2 = xs[i], xs[i+1]
        w = norm_cdf(z(x2)) - norm_cdf(z(x1))
        if ys[i] >= 0 and ys[i+1] >= 0: pop_num += w
        elif ys[i] * ys[i+1] < 0:
            m = x1 - ys[i]*(x2-x1)/(ys[i+1]-ys[i])
            w1 = norm_cdf(z(m)) - norm_cdf(z(x1))
            w2 = norm_cdf(z(x2)) - norm_cdf(z(m))
            if ys[i] >= 0: pop_num += w1
            else: pop_num += w2
        pop_den += w
    pop = pop_num / pop_den if pop_den>0 else None
    return maxP, -minP if minP<0 else 0.0, beL, beH, pop

# ------------------------- Strike selection utilities -------------------------

def width_from_pct(S, pct, stride): return max(stride*2, round_to_stride(pct*S, stride))

def make_leg(kind, side, K, qty, t0, iv0, t_eval, iv_eval) -> Leg:
    return Leg(kind=kind, side=side, K=int(K), qty=int(qty), t0=float(t0), iv0=float(iv0), t_eval=float(t_eval), iv_eval=float(iv_eval))

def choose_strikes(
    mode: str, S: float, em: float, t: float, r: float, q: float, sigma: float, stride: int,
    short_delta: float, wing_delta: float, short_tail_prob: float, wing_tail_prob: float,
    k_sigma: float, width_pct: float
) -> Tuple[int,int,int,int]:
    """
    Return (short_put, long_put, short_call, long_call) integer strikes for a symmetric IC shell.
    - mode='sigma'  : shorts at ± k_sigma*em ; wings wider by width_pct*spot
    - mode='delta'  : shorts at |Δ|=short_delta; wings at |Δ|=wing_delta
    - mode='prob'   : shorts at tail probs; wings at wing tail probs
    """
    if mode == "delta":
        sp = round_to_stride(strike_for_target_put_delta(S, t, r, q, sigma, short_delta), stride)
        lp = round_to_stride(strike_for_target_put_delta(S, t, r, q, sigma, wing_delta),  stride)
        sc = round_to_stride(strike_for_target_call_delta(S, t, r, q, sigma, short_delta), stride)
        lc = round_to_stride(strike_for_target_call_delta(S, t, r, q, sigma, wing_delta),  stride)
        lp = min(lp, sp - max(stride, abs(sp-lp)))
        lc = max(lc, sc + max(stride, abs(lc-sc)))
        return sp, lp, sc, lc

    if mode == "prob":
        sp_f = strike_for_otm_prob(S, t, r, q, sigma, short_tail_prob, is_call=False)
        lp_f = strike_for_otm_prob(S, t, r, q, sigma, wing_tail_prob,  is_call=False)
        sc_f = strike_for_otm_prob(S, t, r, q, sigma, short_tail_prob, is_call=True)
        lc_f = strike_for_otm_prob(S, t, r, q, sigma, wing_tail_prob,  is_call=True)
        sp = round_to_stride(sp_f, stride); lp = round_to_stride(lp_f, stride)
        sc = round_to_stride(sc_f, stride); lc = round_to_stride(lc_f, stride)
        lp = min(lp, sp - max(stride, abs(sp-lp)))
        lc = max(lc, sc + max(stride, abs(lc-sc)))
        return sp, lp, sc, lc

    # default = 'sigma'
    sc = round_to_stride(S + k_sigma*em, stride)
    lc = sc + max(stride*2, round_to_stride(width_pct*S, stride))
    sp = round_to_stride(S - k_sigma*em, stride)
    lp = sp - max(stride*2, round_to_stride(width_pct*S, stride))
    return sp, lp, sc, lc

# ----------------------------- EV optimizer (IC) ------------------------------

def expected_value_of_legs_normal(
    S: float, em: float, r: float, q: float, legs: List[Leg], prem0: float, span_mult: float = 3.0
) -> float:
    """Approx EV under Normal(S, em^2) for payoff at evaluation (not risk-neutral)."""
    lo = max(0.0, S - span_mult*em); hi = S + span_mult*em
    N = 900
    dx = (hi - lo) / N
    ev = 0.0
    denom = em if em>1e-9 else 1.0
    for i in range(N+1):
        x = lo + i*dx
        w = norm_pdf((x - S)/denom) * dx / denom
        ev += pnl_at(x, r, q, legs, prem0) * w
    return ev

def optimize_ic_ev(
    S: float, em: float, tN: float, r: float, q: float, iv: float, stride: int,
    k_list: List[float], w_list: List[float]
) -> Tuple[Tuple[int,int,int,int], float]:
    """Grid over k_sigma and width_pct to maximize EV for Short IC."""
    best = None; best_ev = -1e99
    for k_sigma in (k_list or [1.0, 1.2]):
        for width_pct in (w_list or [0.02]):
            sp, lp, sc, lc = choose_strikes("sigma", S, em, tN, r, q, iv, stride,
                                            0.18,0.05, 0.18,0.05, k_sigma, width_pct)
            legs = [
                make_leg('put','short',sp,1,tN,iv,0.0,iv),
                make_leg('put','long', lp,1,tN,iv,0.0,iv),
                make_leg('call','short',sc,1,tN,iv,0.0,iv),
                make_leg('call','long', lc,1,tN,iv,0.0,iv)
            ]
            p0 = sum(price_leg_entry(S, r, q, L) for L in legs)
            ev = expected_value_of_legs_normal(S, em, r, q, legs, p0)
            if ev > best_ev:
                best_ev = ev; best = (sp, lp, sc, lc)
    return best, best_ev

# ------------------------- Strategy Builders (legs) --------------------------

def build_strategies_full(
    S: float, iv: float, iv2: Optional[float], days_near: int, days_next: Optional[int],
    r: float, q: float, stride: int,
    sigma_ic: float, sigma_bps: float, sigma_ls: float,
    w_ic_pct: float, w_bps_pct: float,
    allow_naked: bool,
    strike_mode: str,
    short_delta: float, wing_delta: float,
    short_tail_prob: float, wing_tail_prob: float,
    ev_k_list: List[float], ev_w_list: List[float]
) -> List[Strat]:

    tN, emN, _ = stride_and_em(S, iv, days_near)
    tF = None
    if iv2 is not None and days_next is not None:
        tF = max(1e-9, days_next/365.0)
    results: List[Strat] = []

    def prem0(legs): return sum(price_leg_entry(S,r,q,L) for L in legs)
    def finish(key,name,group,style,legs,notes):
        p0 = prem0(legs)
        maxP, maxL, beL, beH, pop = scan_metrics(S, emN, r, q, legs, p0)
        roi = None
        if style=="credit" and maxL>0: roi = (p0 if p0>0 else 0.0) / maxL
        elif style=="debit"  and p0<0 and maxP>0: roi = maxP / (-p0)
        results.append(Strat(key,name,group,style,legs,p0,beL,beH,maxP,maxL,roi,pop,notes))

    # --- Iron Condor (Short) with selectable strike logic / EV optimizer ---
    if strike_mode == "ev":
        (sp, lp, sc, lc), _best_ev = optimize_ic_ev(
            S, emN, tN, r, q, iv, stride, k_list=ev_k_list, w_list=ev_w_list
        )
        ic_notes = f"EV-opt · SP {sp}/LP {lp} · SC {sc}/LC {lc}"
    else:
        sp, lp, sc, lc = choose_strikes(
            strike_mode, S, emN, tN, r, q, iv, stride,
            short_delta, wing_delta, short_tail_prob, wing_tail_prob,
            sigma_ic, w_ic_pct
        )
        ic_notes = f"{strike_mode} · SP {sp}/LP {lp} · SC {sc}/LC {lc}"

    legs=[ make_leg('put','short',sp,1,tN,iv,0.0,iv), make_leg('put','long',lp,1,tN,iv,0.0,iv),
           make_leg('call','short',sc,1,tN,iv,0.0,iv), make_leg('call','long',lc,1,tN,iv,0.0,iv) ]
    finish("IC","Short Iron Condor","Neutral","credit",legs,ic_notes)

    # --- Iron Butterfly (Short) wings via same selection logic (ATM body) ---
    atm  = round_to_stride(S, stride)
    if strike_mode == "delta":
        lp_ib = round_to_stride(strike_for_target_put_delta(S, tN, r, q, iv, wing_delta), stride)
        lc_ib = round_to_stride(strike_for_target_call_delta(S, tN, r, q, iv, wing_delta), stride)
    elif strike_mode == "prob":
        lp_ib = round_to_stride(strike_for_otm_prob(S, tN, r, q, iv, wing_tail_prob,  is_call=False), stride)
        lc_ib = round_to_stride(strike_for_otm_prob(S, tN, r, q, iv, wing_tail_prob,  is_call=True),  stride)
    else:
        wIC = width_from_pct(S, w_ic_pct, stride)
        lp_ib = atm - wIC
        lc_ib = atm + wIC

    legs=[ make_leg('put','short',atm,1,tN,iv,0.0,iv), make_leg('put','long',lp_ib,1,tN,iv,0.0,iv),
           make_leg('call','short',atm,1,tN,iv,0.0,iv), make_leg('call','long',lc_ib,1,tN,iv,0.0,iv) ]
    finish("IB","Short Iron Butterfly","Neutral","credit",legs,
           f"{strike_mode} wings · LP {lp_ib} / LC {lc_ib}")

    # --- Short Strangle / Straddle (naked) ---
    if allow_naked:
        kP = round_to_stride(S - 1.1*emN, stride); kC = round_to_stride(S + 1.1*emN, stride)
        legs=[ make_leg('put','short',kP,1,tN,iv,0.0,iv), make_leg('call','short',kC,1,tN,iv,0.0,iv) ]
        finish("SS","Short Strangle","Neutral","credit",legs,"±1.1σ naked")
        legs=[ make_leg('put','short',atm,1,tN,iv,0.0,iv), make_leg('call','short',atm,1,tN,iv,0.0,iv) ]
        finish("SSTRD","Short Straddle","Neutral","credit",legs,"ATM naked")

    # --- Long Strangle / Straddle (sigma-based distance preserved) ---
    kC = round_to_stride(S + sigma_ls*emN, stride); kP = round_to_stride(S - sigma_ls*emN, stride)
    legs=[ make_leg('call','long',kC,1,tN,iv,0.0,iv), make_leg('put','long',kP,1,tN,iv,0.0,iv) ]
    finish("LS","Long Strangle","Neutral","debit",legs,f"±{sigma_ls:.2f}σ long-vol")

    legs=[ make_leg('call','long',atm,1,tN,iv,0.0,iv), make_leg('put','long',atm,1,tN,iv,0.0,iv) ]
    finish("LSTRD","Long Straddle","Neutral","debit",legs,"ATM long-vol")

    # --- Bull Put Spread (credit): optional delta/prob selection ---
    if strike_mode == "delta":
        sp2 = round_to_stride(strike_for_target_put_delta(S,tN,r,q,iv, short_delta), stride)
        lp2 = round_to_stride(strike_for_target_put_delta(S,tN,r,q,iv, max(wing_delta, min(0.5*short_delta, 0.10))), stride)
    elif strike_mode == "prob":
        sp2 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, short_tail_prob, is_call=False), stride)
        lp2 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, wing_tail_prob,  is_call=False), stride)
    else:
        sp2 = round_to_stride(S - 1.0*emN, stride); lp2 = sp2 - width_from_pct(S, w_bps_pct, stride)

    legs=[ make_leg('put','short',sp2,1,tN,iv,0.0,iv), make_leg('put','long',lp2,1,tN,iv,0.0,iv) ]
    finish("BPS","Bull Put Spread","Bullish","credit",legs,"OTM put spread")

    # --- Bull Call Spread (debit): optional delta/prob selection (on calls) ---
    if strike_mode == "delta":
        lc1 = round_to_stride(strike_for_target_call_delta(S,tN,r,q,iv, max(0.30, short_delta)), stride)
        sc1 = round_to_stride(strike_for_target_call_delta(S,tN,r,q,iv, max(0.60, short_delta + 0.25)), stride)
    elif strike_mode == "prob":
        # smaller upper-tail prob for long, larger for short (further OTM)
        lc1 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, max(0.30, short_tail_prob), is_call=True), stride)
        sc1 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, max(0.60, short_tail_prob + 0.25), is_call=True), stride)
    else:
        lc1 = round_to_stride(S + 0.2*emN, stride); sc1 = lc1 + width_from_pct(S, w_bps_pct, stride)

    legs=[ make_leg('call','long',lc1,1,tN,iv,0.0,iv), make_leg('call','short',sc1,1,tN,iv,0.0,iv) ]
    finish("BCS","Bull Call Spread","Bullish","debit",legs,"OTM call spread")

    # --- Bear Call Spread (credit): optional delta/prob selection ---
    if strike_mode == "delta":
        sc2 = round_to_stride(strike_for_target_call_delta(S,tN,r,q,iv, short_delta), stride)
        lc2 = round_to_stride(strike_for_target_call_delta(S,tN,r,q,iv, max(wing_delta, min(0.5*short_delta, 0.10))), stride)
    elif strike_mode == "prob":
        sc2 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, short_tail_prob, is_call=True), stride)
        lc2 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, wing_tail_prob,  is_call=True), stride)
    else:
        sc2 = round_to_stride(S + 1.0*emN, stride); lc2 = sc2 + width_from_pct(S, w_bps_pct, stride)

    legs=[ make_leg('call','short',sc2,1,tN,iv,0.0,iv), make_leg('call','long',lc2,1,tN,iv,0.0,iv) ]
    finish("BECS","Bear Call Spread","Bearish","credit",legs,"OTM call spread")

    # --- Bear Put Spread (debit): optional delta/prob selection ---
    if strike_mode == "delta":
        sp3 = round_to_stride(strike_for_target_put_delta(S,tN,r,q,iv, max(0.30, short_delta)), stride)
        lp3 = round_to_stride(strike_for_target_put_delta(S,tN,r,q,iv, max(0.60, short_delta + 0.25)), stride)
    elif strike_mode == "prob":
        sp3 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, max(0.30, short_tail_prob), is_call=False), stride)
        lp3 = round_to_stride(strike_for_otm_prob(S,tN,r,q,iv, max(0.60, short_tail_prob + 0.25), is_call=False), stride)
    else:
        sp3 = round_to_stride(S - 0.2*emN, stride); lp3 = sp3 - width_from_pct(S, w_bps_pct, stride)

    legs=[ make_leg('put','long',sp3,1,tN,iv,0.0,iv), make_leg('put','short',lp3,1,tN,iv,0.0,iv) ]
    finish("BEPS","Bear Put Spread","Bearish","debit",legs,"OTM put spread")

    # --- Call Ratio Backspread (1 short ATM, 2 long OTM calls) ---
    k_sell = atm; k_buy = round_to_stride(S + 1.0*emN, stride)
    legs=[ make_leg('call','short',k_sell,1,tN,iv,0.0,iv), make_leg('call','long',k_buy,2,tN,iv,0.0,iv) ]
    finish("CRB","Call Ratio Backspread","Bullish","debit",legs,"1× short ATM, 2× long OTM")

    # --- Put Ratio Backspread ---
    k_sellp = atm; k_buyp = round_to_stride(S - 1.0*emN, stride)
    legs=[ make_leg('put','short',k_sellp,1,tN,iv,0.0,iv), make_leg('put','long',k_buyp,2,tN,iv,0.0,iv) ]
    finish("PRB","Put Ratio Backspread","Bearish","debit",legs,"1× short ATM, 2× long OTM")

    # --- Bull/Bear Butterflies (1-2-1) ---
    k2 = round_to_stride(S + 0.5*emN, stride); k1 = k2 - width_from_pct(S, w_bps_pct, stride); k3 = k2 + width_from_pct(S, w_bps_pct, stride)
    legs=[ make_leg('call','long',k1,1,tN,iv,0.0,iv), make_leg('call','short',k2,2,tN,iv,0.0,iv), make_leg('call','long',k3,1,tN,iv,0.0,iv) ]
    finish("BULL_BFLY","Bull Butterfly (calls)","Bullish","debit",legs,"1-2-1")

    k2p = round_to_stride(S - 0.5*emN, stride); k1p = k2p - width_from_pct(S, w_bps_pct, stride); k3p = k2p + width_from_pct(S, w_bps_pct, stride)
    legs=[ make_leg('put','long',k1p,1,tN,iv,0.0,iv), make_leg('put','short',k2p,2,tN,iv,0.0,iv), make_leg('put','long',k3p,1,tN,iv,0.0,iv) ]
    finish("BEAR_BFLY","Bear Butterfly (puts)","Bearish","debit",legs,"1-2-1")

    # --- Bull/Bear Condors (debit; two verticals) ---
    L1 = round_to_stride(S + 0.2*emN, stride); S1 = L1 + width_from_pct(S, w_bps_pct, stride)
    S2 = S1 + width_from_pct(S, w_bps_pct, stride); L2 = S2 + width_from_pct(S, w_bps_pct, stride)
    legs=[ make_leg('call','long',L1,1,tN,iv,0.0,iv), make_leg('call','short',S1,1,tN,iv,0.0,iv),
           make_leg('call','short',S2,1,tN,iv,0.0,iv), make_leg('call','long',L2,1,tN,iv,0.0,iv) ]
    finish("BULL_CONDOR","Bull Condor (calls)","Bullish","debit",legs,"two call spreads")

    pL1 = round_to_stride(S - 0.2*emN, stride); pS1 = pL1 - width_from_pct(S, w_bps_pct, stride)
    pS2 = pS1 - width_from_pct(S, w_bps_pct, stride); pL2 = pS2 - width_from_pct(S, w_bps_pct, stride)
    legs=[ make_leg('put','long',pL1,1,tN,iv,0.0,iv), make_leg('put','short',pS1,1,tN,iv,0.0,iv),
           make_leg('put','short',pS2,1,tN,iv,0.0,iv), make_leg('put','long',pL2,1,tN,iv,0.0,iv) ]
    finish("BEAR_CONDOR","Bear Condor (puts)","Bearish","debit",legs,"two put spreads")

    # --- Long Iron Condor / Long Iron Butterfly (debit versions) ---
    legs=[ make_leg('put','long',atm,1,tN,iv,0.0,iv), make_leg('put','short',lp_ib,1,tN,iv,0.0,iv),
           make_leg('call','long',atm,1,tN,iv,0.0,iv), make_leg('call','short',lc_ib,1,tN,iv,0.0,iv) ]
    finish("LIB","Long Iron Butterfly","Other","debit",legs,"reverse IB")


    # --- Jade Lizard: short put + short call + long higher call (ensure credit ≥ call width) ---
    sp_j = round_to_stride(S - 0.7*emN, stride)
    sc_j = round_to_stride(S + 0.7*emN, stride)
    wSP = width_from_pct(S, w_bps_pct, stride)
    lc_j = sc_j + wSP
    jl_legs=[ make_leg('put','short',sp_j,1,tN,iv,0.0,iv), make_leg('call','short',sc_j,1,tN,iv,0.0,iv), make_leg('call','long',lc_j,1,tN,iv,0.0,iv) ]
    credit = sum(price_leg_entry(S, r, q, L) for L in jl_legs)
    call_width = lc_j - sc_j
    if credit < call_width:
        steps = max(1, int(math.ceil((call_width - credit)/stride)))
        lc_j += steps*stride
        jl_legs[-1] = make_leg('call','long',lc_j,1,tN,iv,0.0,iv)
    finish("JL","Jade Lizard","Bullish","credit",jl_legs,"no upside risk if credit ≥ call width")

    # --- Risk Reversal / Range Forward (near zero-cost hedge) ---
    k_call = round_to_stride(S + 0.6*emN, stride); k_put = round_to_stride(S - 0.6*emN, stride)
    rr_legs=[ make_leg('call','long',k_call,1,tN,iv,0.0,iv), make_leg('put','short',k_put,1,tN,iv,0.0,iv) ]
    finish("RR","Risk Reversal (Range Forward)","Other","other",rr_legs,"near zero-cost directional hedge")

    # --- Strip / Strap (ATM) ---
    strip=[ make_leg('put','long',atm,2,tN,iv,0.0,iv), make_leg('call','long',atm,1,tN,iv,0.0,iv) ]
    finish("STRIP","Strip (2P+1C @ATM)","Other","debit",strip,"bearish long-vol tilt")
    strap=[ make_leg('call','long',atm,2,tN,iv,0.0,iv), make_leg('put','long',atm,1,tN,iv,0.0,iv) ]
    finish("STRAP","Strap (2C+1P @ATM)","Other","debit",strap,"bullish long-vol tilt")

    # --- Batman (two narrow flies around ±d) & Double Plateau (two narrow condors) ---
    d = 0.7*emN; wing = max(suggest_stride(S), stride)
    bat_legs=[]
    for sign in (+1, -1):
        k2b = round_to_stride(S + sign*d, stride); k1b=k2b-wing; k3b=k2b+wing
        bat_legs += [ make_leg('call','long',k1b,1,tN,iv,0.0,iv),
                      make_leg('call','short',k2b,2,tN,iv,0.0,iv),
                      make_leg('call','long',k3b,1,tN,iv,0.0,iv) ]
    finish("BATMAN","Batman (double butterfly)","Neutral","debit",bat_legs,"two flies around ±0.7σ")

    dp_legs=[]
    for sign in (+1, -1):
        mid = round_to_stride(S + sign*d, stride)
        scD = mid; lcD= mid+wing; spD = mid; lpD= mid-wing
        dp_legs += [ make_leg('put','long',lpD,1,tN,iv,0.0,iv), make_leg('put','short',spD,1,tN,iv,0.0,iv),
                     make_leg('call','short',scD,1,tN,iv,0.0,iv), make_leg('call','long',lcD,1,tN,iv,0.0,iv) ]
    finish("DPLAT","Double Plateau (double condor)","Neutral","debit",dp_legs,"two condors around ±0.7σ")

    # --- Calendars (need next expiry & iv2) – evaluated at NEAR expiry ---
    if tF and iv2:
        rem = max(1e-9, tF - tN)
        legs=[ make_leg('call','long',atm,1,tF,iv2,rem,iv2), make_leg('call','short',atm,1,tN,iv,0.0,iv) ]
        finish("LCC","Long Calendar (Calls)","Neutral","debit",legs,"value @ near-expiry incl. far-call time value")
        legs=[ make_leg('put','long',atm,1,tF,iv2,rem,iv2), make_leg('put','short',atm,1,tN,iv,0.0,iv) ]
        finish("LCP","Long Calendar (Puts)","Neutral","debit",legs,"value @ near-expiry incl. far-put time value")

    return results

# ------------------------------ Ranking --------------------------------------

def score_strategy(s: Strat, S: float, em: float) -> float:
    pop = s.pop if s.pop is not None else 0.5
    roi = s.roi if s.roi is not None else 0.0
    span = 0.0
    if s.be_low is not None and s.be_high is not None:
        span = max(0.0, (s.be_high - s.be_low)/(2*em + 1e-9))
    elif s.be_low is not None:  span = max(0.0, (S - s.be_low)/(2*em + 1e-9))
    elif s.be_high is not None: span = max(0.0, (s.be_high - S)/(2*em + 1e-9))
    uncapped_credit = (s.style=="credit" and (s.max_loss_sh==0.0 or s.max_loss_sh is None))
    penalty = 0.15 if uncapped_credit else 0.0
    return 0.55*pop + 0.30*min(max(roi,0.0),2.0) + 0.20*min(span,1.0) - penalty

# ------------------------------ HTML -----------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IV Agent – Top Strategies</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{--bg:#070b16;--card:#0f1629;--panel:#0b1222;--txt:#e8ecf6;--muted:#a4afc6;--line:#1b2a4a;--accent:#7aa2f7;--ok:#16a34a;--warn:#f59e0b}
  *{box-sizing:border-box}
  body{margin:0;background:
      radial-gradient(1000px 600px at 10% -10%,#1c2550 0%,transparent 60%),
      radial-gradient(900px 600px at 110% 10%,#1d3a3f 0%,transparent 55%),var(--bg);
      color:var(--txt);font-family:system-ui,-apple-system,Segoe UI,Roboto,Inter,Ubuntu,Arial,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:24px}
  h1{margin:0 0 4px 0;font-size:28px;letter-spacing:.3px}
  .meta{color:var(--muted);margin-bottom:18px}
  .toolbar{display:flex;gap:10px;align-items:center;margin:6px 0 20px 0}
  .search{flex:1;max-width:420px;background:#0b1427;border:1px solid #1d2a49;color:var(--txt);
          border-radius:12px;padding:10px 12px;outline:none}
  .btn{background:var(--accent);color:#fff;border:none;border-radius:12px;padding:10px 14px;cursor:pointer;text-decoration:none}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
  .card{background:linear-gradient(180deg,#111b33 0%,#0d1428 100%);border:1px solid #1a2747;border-radius:16px;padding:14px;cursor:pointer;transition:transform .12s ease,border-color .12s}
  .card:hover{transform:translateY(-2px);border-color:#2a3a6a}
  .sym{font-weight:700;font-size:18px;letter-spacing:.4px}
  .row{display:flex;justify-content:space-between;margin-top:8px;color:var(--muted)}
  .pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-top:10px;background:#173225;color:#8bffa9}

  /* Detail modal */
  .detail{position:fixed;inset:0;background:rgba(5,8,15,.65);display:none;align-items:flex-start;justify-content:center;padding:32px 16px;backdrop-filter:blur(4px);z-index:9999}
  .panel{width:min(1280px,96vw);max-height:92vh;overflow:auto;background:var(--panel);border:1px solid #1c294b;border-radius:18px;padding:22px}
  .panelhead{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-bottom:12px}
  .title{font-size:24px;font-weight:800}
  .sub{color:var(--muted);font-size:14px}
  .close{background:#25365e;border:1px solid #334876;color:#dbe4ff;border-radius:10px;padding:8px 10px;cursor:pointer}

  /* Layout: left = strategy list; right = strategy detail; chart comes AFTER details */
  .cols{display:grid;grid-template-columns:0.95fr 1.05fr;gap:18px}
  @media (max-width:1024px){.cols{grid-template-columns:1fr}}
  .box{background:#0e1933;border:1px solid #1c2a51;border-radius:14px;padding:14px}

  /* Strategy list (user friendly) */
  .slist{display:flex;flex-direction:column;gap:8px;max-height:540px;overflow:auto;padding-right:4px}
  .sitem{display:flex;align-items:center;justify-content:space-between;gap:10px;background:#0f1835;border:1px solid #233463;border-radius:12px;padding:10px 12px;cursor:pointer}
  .sitem:hover{border-color:#2f4aa8}
  .sitem.active{outline:2px solid #2f4aa8}
  .sleft{display:flex;align-items:center;gap:10px;min-width:0}
  .dot{width:10px;height:10px;border-radius:50%}
  .sname{font-weight:700;font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
  .stags{color:#a5b4fc;font-size:12px}
  .sright{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .badge{font-size:12px;border-radius:999px;padding:3px 8px;border:1px solid #334876;background:#162243}
  .badge.ok{border-color:#285a36;background:#173225;color:#8bffa9}
  .badge.warn{border-color:#7a540a;background:#382c0b;color:#ffd88a}

  /* Selected strategy detail bigger */
  #selName{font-size:20px;font-weight:800;margin-bottom:8px}
  #ordersBody{font-size:15px;line-height:1.5;background:#0b1427;border:1px solid #1b2a4a;border-radius:12px;padding:12px;white-space:pre-wrap}
  table{width:100%;border-collapse:collapse;font-size:15px;margin-top:8px}
  th,td{padding:9px;border-bottom:1px solid #1e2846;text-align:left}
  .rowbtns{display:flex;gap:8px;margin-top:10px}

  /* Chart AFTER details */
  canvas{width:100%;height:380px;background:#0b1427;border:1px solid #1b2a4a;border-radius:14px}
  .legend{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}
  .chip{display:flex;align-items:center;gap:6px;background:#162243;border:1px solid #233463;border-radius:999px;padding:8px 12px;cursor:pointer;font-size:14px}
  .chip.active{outline:2px solid #2f4aa8}
  .tab{display:inline-block;margin:0 8px 8px 0;padding:8px 12px;border-radius:999px;background:#14203f;border:1px solid #233463;cursor:pointer}
  .tab.active{outline:2px solid #2f4aa8}
</style>
</head>
<body>
<div class="wrap">
  <h1>IV Agent – Top Strategies</h1>
  <div class="meta">Generated: __GENERATED__ · Horizon: __DAYS__ days · Data: NSE · <a class="btn" href="__CSV_NAME__" download>CSV</a></div>
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
      <!-- LEFT: Recommended list -->
      <div class="box">
        <div style="margin-bottom:8px;color:#cbd5e1"><b>Recommended (best first)</b> · click to view details</div>
        <div id="strategyList" class="slist"></div>
      </div>

      <!-- RIGHT: Details first -->
      <div>
        <div class="box">
          <div style="margin-bottom:8px;color:#cbd5e1"><b>Selected strategy</b></div>
          <div id="selName">—</div>

          <div style="margin:8px 0;color:#cbd5e1"><b>Orders (per lot)</b></div>
          <pre id="ordersBody">—</pre>

          <table id="tbl"><thead>
            <tr><th>Credit/Debit</th><th>B/E</th><th>Max P/L (₹/lot)</th><th>ROI%</th><th>POP%</th></tr>
          </thead><tbody></tbody></table>

          <div style="margin-top:12px;color:#cbd5e1"><b>Greeks @ Spot (per lot)</b></div>
          <table id="greeksTbl"><thead>
            <tr><th>Δ</th><th>Γ</th><th>Θ (per day)</th><th>Vega</th></tr>
          </thead><tbody><tr><td colspan="4">—</td></tr></tbody></table>

          <div style="margin-top:12px;color:#cbd5e1"><b>P&L Table (₹/lot)</b> · key points</div>
          <table id="pnlTbl"><thead>
            <tr><th>Price</th><th>Label</th><th>Payoff</th></tr>
          </thead><tbody><tr><td colspan="3">—</td></tr></tbody></table>

          <div class="rowbtns">
            <button id="dlPnl" class="btn">Download P&L CSV</button>
            <button id="dlGreeks" class="btn">Download Greeks CSV</button>
          </div>
        </div>

        <!-- THEN: payoff chart + legend -->
        <div class="box" style="margin-top:14px">
          <div id="tabs"></div>
          <canvas id="chart" width="1000" height="380"></canvas>
          <div class="legend" id="legend"></div>
          <div style="margin-top:8px"><span id="toggleAll" class="tab">Show all strategies</span></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
/* ---------- Polyfill for Math.erf (some browsers miss it) ---------- */
if (typeof Math.erf !== 'function') {
  Math.erf = function(x){
    const sign = (x>=0)? 1 : -1;
    x = Math.abs(x);
    const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
    const t=1/(1+p*x);
    const y=1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
    return sign*y;
  }
}
function cdf(x){return 0.5*(1+Math.erf(x/Math.SQRT2));}
function bsPrice(S,K,t,r,sigma,q,type){
  if(t<=1e-9) return Math.max(0,type==='call'?S-K:K-S);
  const d1=(Math.log(S/K)+(r-q+0.5*sigma*sigma)*t)/(sigma*Math.sqrt(t));
  const d2=d1-sigma*Math.sqrt(t);
  if(type==='call') return S*Math.exp(-q*t)*cdf(d1)-K*Math.exp(-r*t)*cdf(d2);
  return K*Math.exp(-r*t)*cdf(-d2)-S*Math.exp(-q*t)*cdf(-d1);
}
/* Greeks */
function bsGreeks(S,K,t,r,sigma,q,type){
  if (t<=1e-9) {
    const intrinsic = (type==='call') ? (S>K) : (K>S);
    const delta = (type==='call') ? (intrinsic? 1: 0) : (intrinsic? -1: 0);
    return {delta, gamma:0, theta_per_year:0, vega:0};
  }
  const d1=(Math.log(S/K)+(r-q+0.5*sigma*sigma)*t)/(sigma*Math.sqrt(t));
  const d2=d1-sigma*Math.sqrt(t);
  const pdf = Math.exp(-0.5*d1*d1)/Math.sqrt(2*Math.PI);
  const Nd1 = cdf(d1), Nd2 = cdf(d2);
  let delta;
  if (type==='call') delta = Math.exp(-q*t)*Nd1;
  else delta = Math.exp(-q*t)*(Nd1-1);
  const gamma = (Math.exp(-q*t)*pdf)/(S*sigma*Math.sqrt(t));
  const theta = -(S*Math.exp(-q*t)*pdf*sigma)/(2*Math.sqrt(t))
              - (type==='call'
                   ? ( -q*S*Math.exp(-q*t)*Nd1 + r*K*Math.exp(-r*t)*Nd2 )
                   : ( -q*S*Math.exp(-q*t)*(Nd1-1) - r*K*Math.exp(-r*t)*cdf(-d2) ));
  const vega  = S*Math.exp(-q*t)*pdf*Math.sqrt(t);
  return {delta, gamma, theta_per_year:theta, vega};
}

function legText(L){
  const side = L.side==='long' ? 'BUY' : 'SELL';
  const kind = L.kind.toUpperCase();
  const qty  = (L.qty||1)+'×';
  const exp  = (L.t_eval>0 ? 'FAR' : 'NEAR');
  return `${side} ${qty} ${exp} ${kind} ${L.K}`;
}

function payoffFromLegs(ST, legs, prem0){
  const r=__R__; const q=__Q__;
  let val=0;
  legs.forEach(L=>{
    const sign = L.side==='long'? 1 : -1;
    const t=L.t_eval; const K=L.K; const iv=L.iv_eval;
    let v=0;
    if(t<=1e-9){ v = L.kind==='call' ? Math.max(0, ST-K) : Math.max(0, K-ST); }
    else{ v = bsPrice(ST,K,t,r,iv,q,L.kind); }
    val += sign * L.qty * v;
  });
  return (val - prem0) * CUR.lot;
}

function aggregateGreeksPerLot(S, r, q, strat, lot){
  let agg = {delta:0, gamma:0, theta_per_year:0, vega:0};
  strat.legs.forEach(L=>{
    const sign = (L.side==='long')? 1 : -1;
    const g = bsGreeks(S, L.K, Math.max(0,L.t_eval), r, Math.max(1e-9,L.iv_eval), q, L.kind);
    agg.delta += sign * L.qty * g.delta;
    agg.gamma += sign * L.qty * g.gamma;
    agg.theta_per_year += sign * L.qty * g.theta_per_year;
    agg.vega  += sign * L.qty * g.vega;
  });
  Object.keys(agg).forEach(k=> agg[k]*=lot );
  return agg;
}

function formatGreeksRow(g){
  const theta_per_day = g.theta_per_year/365.0;
  return [ g.delta.toFixed(3), g.gamma.toExponential(3), theta_per_day.toFixed(2), g.vega.toFixed(2) ];
}

function keyPointsForPnL(S, em, strat){
  const pts = [
    {x:S-2*em, label:'S−2σ'},
    {x:S-1*em, label:'S−1σ'},
    {x:S,      label:'Spot'},
    {x:S+1*em, label:'S+1σ'},
    {x:S+2*em, label:'S+2σ'}
  ];
  if (strat.be_low!=null)  pts.push({x:strat.be_low,  label:'BE low'});
  if (strat.be_high!=null) pts.push({x:strat.be_high, label:'BE high'});
  pts.sort((a,b)=>a.x-b.x);
  const uniq=[]; let last=null;
  pts.forEach(p=>{ if(last==null || Math.abs(p.x-last.x) > Math.max(1, S*0.0001)){ uniq.push(p); last=p;}});
  return uniq;
}

function legsToSummary(legs){ return legs.map(legText).join(', '); }

/* ---------- Data ---------- */
const DATA = __DATA_JSON__;
const TOPN = __TOPN__;
const cards = document.getElementById('cards');
document.getElementById('q').addEventListener('input', e=>{
  const t=e.target.value.toLowerCase();
  Array.from(cards.children).forEach(c=>{c.style.display=c.textContent.toLowerCase().includes(t)?'':''})
});
DATA.forEach((d,i)=>{
  const card=document.createElement('div');card.className='card';card.onclick=()=>openDetail(i);
  const ivStr = (d.iv!=null && isFinite(d.iv)) ? (d.iv*100).toFixed(1)+'%' : '—';
  const pillText = (d.top && d.top.length) ? d.top.map(x=>x.name).join(' · ') : 'No strategies';
  card.innerHTML=`<div class="sym">${d.symbol}</div>
  <div class="row"><span>Spot</span><span>₹${(+d.spot).toFixed(2)}</span></div>
  <div class="row"><span>IV (near)</span><span>${ivStr}</span></div>`;
  const p=document.createElement('div');p.className='pill';p.textContent=pillText;
  card.appendChild(p); cards.appendChild(card);
});

/* ---------- Panel refs ---------- */
const detail=document.getElementById('detail');
const d_sym=document.getElementById('d_sym'); const d_meta=document.getElementById('d_meta');
const tblBody = document.querySelector('#tbl tbody');
const legend=document.getElementById('legend'); const tabs=document.getElementById('tabs');
const toggleAll=document.getElementById('toggleAll');
const strategyList = document.getElementById('strategyList');
const ordersBox = document.getElementById('ordersBody');
const selName = document.getElementById('selName');
const greeksBody = document.querySelector('#greeksTbl tbody');
const pnlBody = document.querySelector('#pnlTbl tbody');
const dlPnlBtn = document.getElementById('dlPnl');
const dlGreeksBtn = document.getElementById('dlGreeks');

let CUR=null, SHOW_ALL=false, selKeys=[], activeKey=null;

function openDetail(i){
  CUR=DATA[i];
  if (!CUR || !Array.isArray(CUR.all) || CUR.all.length===0) {
    d_sym.textContent = CUR?.symbol || '—';
    d_meta.textContent=`Lot: ${CUR?.lot||'—'} · Spot: ₹${(CUR?.spot||0).toFixed(2)} · No strategies computed`;
    strategyList.innerHTML = '';
    legend.innerHTML = '';
    tblBody.innerHTML = '<tr><td colspan="5">No strategies available for this symbol.</td></tr>';
    greeksBody.innerHTML = '<tr><td colspan="4">—</td></tr>';
    pnlBody.innerHTML = '<tr><td colspan="3">—</td></tr>';
    ordersBox.textContent = '—';
    detail.style.display='flex';
    renderTabs(); draw(); 
    return;
  }
  const bestKey = (CUR.top && CUR.top[0]) ? CUR.top[0].key : CUR.all[0].key;
  activeKey = bestKey; selKeys=[bestKey]; SHOW_ALL=false;

  d_sym.textContent=CUR.symbol;
  const ivx = (CUR.iv2!=null && isFinite(CUR.iv2)) ? (' · IVx '+(CUR.iv2*100).toFixed(1)+'%') : '';
  d_meta.textContent=`Expiry near: ${CUR.exp_near||'—'} · next: ${CUR.exp_next||'—'} · Lot: ${CUR.lot} · Spot: ₹${(+CUR.spot).toFixed(2)} · IVn ${(CUR.iv*100).toFixed(1)}%${ivx}`;

  renderStrategyList();
  renderTabs();
  renderLegend();
  updateSelectedDetail();
  draw();
  detail.style.display='flex';
}
function closeDetail(){ detail.style.display='none'; }

function colorFor(i){return ['#60a5fa','#34d399','#fbbf24','#f472b6','#22d3ee','#e5e7eb'][i%6];}
function allStratsBestFirst(){
  const best = (CUR.top && CUR.top[0]) ? CUR.top[0] : CUR.all[0];
  const rest = CUR.all.filter(s=>s.key!==best.key);
  return [best, ...rest];
}

/* --------- helper badges --------- */
function styleEmoji(style){
  if (style==='credit') return '💰';
  if (style==='debit')  return '💸';
  return '🧭';
}
function beSpanInfo(s){
  if (s.be_low!=null && s.be_high!=null && isFinite(s.be_low) && isFinite(s.be_high) && CUR && isFinite(CUR.em) && CUR.em>0){
    const span = (s.be_high - s.be_low)/(2*CUR.em);
    return {label: 'Width '+span.toFixed(2)+'σ', val: span};
  }
  if (s.be_low!=null || s.be_high!=null) return {label: '1-side BE', val: 0.4};
  return {label: 'No BE', val: 0.0};
}

/* --------- Strategy list renderer --------- */
function renderStrategyList(){
  strategyList.innerHTML='';
  const arr = allStratsBestFirst();
  arr.forEach((s,idx)=>{
    const item=document.createElement('div'); item.className='sitem'+(activeKey===s.key?' active':'');
    const left=document.createElement('div'); left.className='sleft';
    const dot=document.createElement('span'); dot.className='dot'; dot.style.background=colorFor(idx);
    const sname=document.createElement('div'); sname.className='sname'; sname.textContent=`${styleEmoji(s.style)} ${s.name}`;
    const stags=document.createElement('div'); stags.className='stags'; stags.textContent=s.group+' · '+(s.style||'');
    left.appendChild(dot); left.appendChild(sname); left.appendChild(stags);

    const right=document.createElement('div'); right.className='sright';
    const pop=document.createElement('span'); pop.className='badge '+((s.pop||0)>=0.6?'ok':((s.pop||0)>=0.5?'':'warn'));
    pop.textContent = 'POP ' + (s.pop==null?'—':(s.pop*100).toFixed(0)) + '%';
    const roi=document.createElement('span'); roi.className='badge'; roi.textContent='ROI ' + (s.roi==null?'—':(s.roi*100).toFixed(0)) + '%';

    const spanInfo = beSpanInfo(s);
    const spanBadge=document.createElement('span');
    spanBadge.className='badge ' + (spanInfo.val>=1.0 ? 'ok' : (spanInfo.val<0.6 ? 'warn' : ''));
    spanBadge.textContent = spanInfo.label;

    right.appendChild(pop); right.appendChild(roi); right.appendChild(spanBadge);

    item.appendChild(left); item.appendChild(right);
    item.onclick=()=>{ activeKey=s.key; selKeys=[s.key]; SHOW_ALL=false; 
      Array.from(strategyList.children).forEach(el=>el.classList.remove('active')); item.classList.add('active');
      renderLegend(); updateSelectedDetail(); draw(); 
    };
    strategyList.appendChild(item);
  });
}

/* --------- Legend & toggles --------- */
function renderLegend(){
  legend.innerHTML='';
  const base = SHOW_ALL ? CUR.all : allStratsBestFirst();
  base.forEach((s,i)=>{
    const chip=document.createElement('span');chip.className='chip'+(selKeys.includes(s.key)?' active':'');
    chip.innerHTML=`<span style="width:10px;height:10px;border-radius:50%;background:${colorFor(i)}"></span>${styleEmoji(s.style)} ${s.name}`;
    chip.onclick=()=>{ 
      if(selKeys.includes(s.key)) selKeys=selKeys.filter(k=>k!==s.key);
      else selKeys.push(s.key);
      if (!selKeys.length) selKeys=[base[0].key];
      activeKey = selKeys[0];
      updateSelectedDetail(); draw(); renderLegend();
    };
    legend.appendChild(chip);
  });
  toggleAll.textContent = SHOW_ALL ? 'Show only best/selected' : 'Show all strategies';
  toggleAll.onclick=()=>{ SHOW_ALL=!SHOW_ALL; renderLegend(); draw(); };
}

function fmt(x){return x==null?'—':(typeof x==='number'?x.toFixed(2):x);}

function updateSelectedDetail(){
  const s = CUR.all.find(x=>x.key===activeKey) || (CUR.top && CUR.top[0]) || CUR.all[0];
  if (!s){ return; }

  selName.textContent = `${styleEmoji(s.style)} ${s.name}`;

  const lines = s.legs.map(L=>legText(L));
  const cdSh = s.prem0_sh;
  const cdLot = cdSh * CUR.lot;
  ordersBox.textContent = `• ${s.name}\n  ${lines.join('\n  ')}\n  Net premium (₹/sh): ${(cdSh>=0?'+':'')}${cdSh.toFixed(2)}  |  (₹/lot): ${(cdLot).toFixed(2)}`;

  const be = fmt(s.be_low)+' / '+fmt(s.be_high);
  const maxp_lot = s.max_profit_sh==null ? 'Uncapped' : (s.max_profit_sh*CUR.lot).toFixed(2);
  const maxl_lot = s.max_loss_sh==null ? '—' : (s.max_loss_sh*CUR.lot).toFixed(2);
  const roi = s.roi==null? '—' : (s.roi*100).toFixed(1);
  const pop = s.pop==null? '—' : (s.pop*100).toFixed(1);
  tblBody.innerHTML='';
  const tr=document.createElement('tr');
  tr.innerHTML = `<td>${(cdSh>=0?'+':'')+cdSh.toFixed(2)} / ${cdLot.toFixed(2)}</td><td>${be}</td><td>${maxp_lot} · ${maxl_lot}</td><td>${roi}</td><td>${pop}</td>`;
  tblBody.appendChild(tr);

  const g = aggregateGreeksPerLot(CUR.spot, __R__, __Q__, s, CUR.lot);
  const gRow = formatGreeksRow(g);
  greeksBody.innerHTML = `<tr><td>${gRow[0]}</td><td>${gRow[1]}</td><td>${gRow[2]}</td><td>${gRow[3]}</td></tr>`;

  const pts = keyPointsForPnL(CUR.spot, CUR.em, s);
  const f = (ST)=>payoffFromLegs(ST, s.legs, s.prem0_sh);
  pnlBody.innerHTML='';
  pts.forEach(p=>{
    const v = f(p.x);
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>₹${p.x.toFixed(2)}</td><td>${p.label}</td><td>${v.toFixed(2)}</td>`;
    pnlBody.appendChild(tr);
  });

  dlPnlBtn.onclick = ()=>{
    const rows=[['Price','Label','Payoff (₹/lot)']];
    pts.forEach(p=>rows.push([p.x.toFixed(2), p.label, f(p.x).toFixed(2)]));
    downloadCSV(rows, `${CUR.symbol}_${s.key}_pnl.csv`);
  };
  dlGreeksBtn.onclick = ()=>{
    const rows=[['Delta','Gamma','Theta/day','Vega']];
    rows.push(gRow);
    downloadCSV(rows, `${CUR.symbol}_${s.key}_greeks.csv`);
  };
}

function renderTabs(){
  tabs.innerHTML='';
  ['±1σ','±1.5σ','±2σ'].forEach((t,idx)=>{
    const span=document.createElement('span');span.className='tab'+(idx==0?' active':'');span.textContent=t;span.dataset.m=[1,1.5,2][idx];
    span.onclick=(e)=>{Array.from(tabs.children).forEach(c=>c.classList.remove('active')); e.target.classList.add('active'); draw(parseFloat(e.target.dataset.m));}
    tabs.appendChild(span);
  });
}

function draw(mult=1.0){
  const c=document.getElementById('chart'); const ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  const source = SHOW_ALL ? CUR.all : CUR.all.filter(s=>selKeys.includes(s.key));
  if(!source || !source.length){
    ctx.strokeStyle="#1b2a4a"; ctx.strokeRect(40,20,c.width-80,c.height-50);
    const y0=c.height-30; ctx.beginPath(); ctx.moveTo(40,y0); ctx.lineTo(c.width-40,y0); ctx.stroke();
    return;
  }
  const S=+CUR.spot, em=+CUR.em, range=mult*em; const xMin=Math.max(0,S-range*1.1), xMax=S+range*1.1;
  const N=340, xs=[], series=source.map(()=>[]);
  for(let i=0;i<=N;i++){
    const ST=xMin+(xMax-xMin)*i/N; xs.push(ST);
    source.forEach((s,idx)=>{ series[idx].push(payoffFromLegs(ST,s.legs,s.prem0_sh)); });
  }
  let yMin=0,yMax=0; series.forEach(a=>a.forEach(v=>{yMin=Math.min(yMin,v); yMax=Math.max(yMax,v)}));
  if(yMin===yMax){yMin-=1;yMax+=1} const pad=(yMax-yMin)*0.12; yMin-=pad; yMax+=pad;
  const X=x=>((x-xMin)/(xMax-xMin))*(c.width-60)+40; const Y=y=>c.height-30 - ((y-yMin)/(yMax-yMin))*(c.height-60);
  ctx.strokeStyle="#1b2a4a"; ctx.strokeRect(40,20,c.width-80,c.height-50);
  const xS=X(S); ctx.beginPath(); ctx.moveTo(xS,20); ctx.lineTo(xS,c.height-30); ctx.stroke(); const y0=Y(0); ctx.beginPath(); ctx.moveTo(40,y0); ctx.lineTo(c.width-40,y0); ctx.stroke();
  ctx.fillStyle="#a4afc6"; ctx.font="12px Segoe UI, Roboto"; ctx.fillText(`S=₹${S.toFixed(2)}`, xS+4, 30);

  source.forEach((s,idx)=>{
    const col=['#60a5fa','#34d399','#fbbf24','#f472b6','#22d3ee','#e5e7eb'][idx%6];
    ctx.beginPath(); ctx.strokeStyle=col; ctx.lineWidth=2;
    series[idx].forEach((v,i)=>{ const px=X(xs[i]), py=Y(v); if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py); });
    ctx.stroke();
  });
}

/* CSV download helper */
function downloadCSV(rows, filename){
  const csv = rows.map(r=>r.map(v=>{
    const s = (v==null)?'':String(v);
    return /[",\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s;
  }).join(',')).join('\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const link = document.createElement('a'); link.href=URL.createObjectURL(blob); link.download=filename;
  document.body.appendChild(link); link.click(); document.body.removeChild(link);
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

def leg_to_text(leg: Leg) -> str:
    exp = "FAR" if leg.t_eval > 0 else "NEAR"
    side = "BUY" if leg.side == "long" else "SELL"
    kind = leg.kind.upper()
    qty  = f"{leg.qty}×"
    return f"{side} {qty} {exp} {kind} {leg.K}"

def legs_summary(legs: List[Leg]) -> str:
    return "; ".join(leg_to_text(L) for L in legs)

# ------------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="IV Agent – Full strategy set with ranking + website")
    ap.add_argument("--tickers", required=True, type=str)
    ap.add_argument("--days", type=int, default=30)  # target for NEAR expiry
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

    # NEW: strike selection knobs
    ap.add_argument("--strike_mode", type=str, default="sigma",
                    choices=["sigma","delta","prob","ev"],
                    help="sigma=±k·σ; delta=target |Δ|; prob=tail probs; ev=optimize IC EV")
    ap.add_argument("--short_delta", type=float, default=0.18, help="|Δ| for short strikes (delta mode)")
    ap.add_argument("--wing_delta",  type=float, default=0.05, help="|Δ| for long wings (delta mode)")
    ap.add_argument("--short_tail_prob", type=float, default=0.18, help="Tail prob for shorts (prob mode)")
    ap.add_argument("--wing_tail_prob",  type=float, default=0.05, help="Tail prob for wings (prob mode)")
    ap.add_argument("--ev_k_sigma_grid", type=str, default="0.8,1.0,1.2,1.4",
                    help="CSV of k multipliers for short strikes relative to σ (ev mode)")
    ap.add_argument("--ev_width_pct_grid", type=str, default="0.01,0.015,0.02,0.025",
                    help="CSV of widths (as % of spot) for condor wings (ev mode)")

    args = ap.parse_args()

    # Parse EV grids
    ev_k_list = [float(x) for x in (args.ev_k_sigma_grid or "").split(",") if x.strip()]
    ev_w_list = [float(x) for x in (args.ev_width_pct_grid or "").split(",") if x.strip()]

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    lots = fetch_lot_sizes()

    # CSV overrides if provided
    spot_map = read_csv_map(args.spot_csv, "ticker", "spot") if args.spot_csv else {}
    iv_map   = read_csv_map(args.iv_csv,   "ticker", "iv")   if args.iv_csv   else {}

    rows_csv: List[Dict[str,str]] = []
    js_rows: List[dict] = []

    for sym in tickers:
        lot = lots.get(normalize_symbol_key(sym), 1)

        spot=None; exp_near=None; exp_next=None; iv_near=None; iv_next=None
        if args.live_iv:
            spot, exp_near, exp_next, iv_near, iv_next = fetch_live(sym, args.days)
        if spot is None: spot = spot_map.get(sym)
        if iv_near is None: iv_near = iv_map.get(sym, args.default_iv)
        if iv_near and iv_near>1.0: iv_near/=100.0
        if iv_next and iv_next>1.0: iv_next/=100.0
        if spot is None: 
            if args.use_yfinance_fallback:
                try:
                    import yfinance as yf
                    s = sym if sym.endswith(".NS") else f"{sym}.NS"
                    data = yf.Ticker(s).history(period="1d")
                    if data is not None and not data.empty:
                        spot = float(data["Close"].iloc[-1])
                except Exception: pass
        if spot is None: 
            if args.verbose: print(f"[SKIP] {sym}: no spot")
            continue

        # times
        days_near = args.days
        days_next = None
        if exp_near and exp_next:
            d1 = dt.datetime.strptime(exp_near, "%d-%b-%Y").date()
            d2 = dt.datetime.strptime(exp_next, "%d-%b-%Y").date()
            days_near = max(1, (d1 - dt.date.today()).days)
            days_next = max(days_near+1, (d2 - dt.date.today()).days)

        tN, emN, stride = stride_and_em(spot, iv_near, days_near)

        # Build ALL strategies
        all_strats = build_strategies_full(
            spot, iv_near, iv_next, days_near, days_next,
            args.risk_free, args.yield_div, stride,
            args.sigma_ic, args.sigma_bps, args.sigma_ls,
            args.width_pct_ic, args.width_pct_bps,
            args.allow_naked,
            args.strike_mode,
            args.short_delta, args.wing_delta,
            args.short_tail_prob, args.wing_tail_prob,
            ev_k_list, ev_w_list
        )

        # Score & pick top N
        scored = [(score_strategy(s, spot, emN), s) for s in all_strats]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _,s in scored[:max(1,args.top_n)]]

        # CSV rows: one row per TopN strategy with strikes + per-lot economics
        def fmt2(x): return "" if x is None else f"{x:.2f}"
        for rank, s in enumerate(top, start=1):
            prem_lot = s.prem0_sh * lot
            maxp_lot = None if s.max_profit_sh is None else s.max_profit_sh * lot
            maxl_lot = None if s.max_loss_sh is None else s.max_loss_sh * lot
            pnl_at_spot_per_sh = pnl_at(spot, args.risk_free, args.yield_div, s.legs, s.prem0_sh)
            pnl_at_spot_lot = pnl_at_spot_per_sh * lot

            rows_csv.append({
                "Symbol": sym,
                "Rank": rank,
                "Strategy": s.name,
                "Lot": lot,
                "Spot": f"{spot:.2f}",
                "IV_near": f"{iv_near:.4f}",
                "Expiry_near": exp_near or "",
                "Expiry_next": exp_next or "",
                "1σ Move(₹)": f"{emN:.2f}",
                "Range ±1σ": f"{spot-emN:.2f}–{spot+emN:.2f}",
                "POP%": "" if s.pop is None else f"{s.pop*100:.1f}",
                "ROI%": "" if s.roi is None else f"{s.roi*100:.1f}",
                "BE_low": fmt2(s.be_low),
                "BE_high": fmt2(s.be_high),
                "Legs / Strikes": legs_summary(s.legs),
                "Net Premium (₹/sh)": f"{s.prem0_sh:.2f}",
                "Net Premium (₹/lot)": f"{prem_lot:.2f}",
                "Max Profit (₹/lot)": "Uncapped" if maxp_lot is None else f"{maxp_lot:.2f}",
                "Max Loss (₹/lot)": "" if maxl_lot is None else f"{maxl_lot:.2f}",
                "PnL @ Spot (₹/lot)": f"{pnl_at_spot_lot:.2f}",
                "StrikeMode": getattr(args, "strike_mode", "sigma"),
            })

        # Website payload
        def legs_to_js(legs: List[Leg]):
            return [ {"kind":L.kind,"side":L.side,"K":L.K,"qty":L.qty,"t_eval":L.t_eval,"iv_eval":L.iv_eval} for L in legs ]
        def strat_to_js(s: Strat):
            return {"key":s.key,"name":s.name,"group":s.group,"style":s.style,"legs":legs_to_js(s.legs),
                    "prem0_sh":float(f"{s.prem0_sh:.6f}"),"be_low":s.be_low,"be_high":s.be_high,
                    "max_profit_sh":s.max_profit_sh,"max_loss_sh":s.max_loss_sh,
                    "roi":s.roi,"pop":s.pop,"notes":s.notes}
        js_rows.append({
            "symbol": sym, "lot": int(lot), "spot": float(f"{spot:.6f}"),
            "iv": float(f"{iv_near:.6f}"), "iv2": None if iv_next is None else float(f"{iv_next:.6f}"),
            "exp_near": exp_near or "", "exp_next": exp_next or "",
            "days": days_near, "em": float(f"{emN:.6f}"),
            "top": [strat_to_js(s) for s in top],
            "all": [strat_to_js(s) for _,s in scored]
        })

    if not js_rows:
        print("No data generated. Check symbols/connectivity."); sys.exit(2)

    write_csv(args.out_csv, rows_csv)

    html = (HTML
            .replace("__GENERATED__", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            .replace("__DAYS__", str(args.days))
            .replace("__CSV_NAME__", os.path.basename(args.out_csv))
            .replace("__TOPN__", str(max(1,args.top_n)))
            .replace("__R__", str(args.risk_free))
            .replace("__Q__", str(args.yield_div))
            .replace("__DATA_JSON__", json.dumps(js_rows, ensure_ascii=False))
            )
    with open(args.html_out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved CSV : {os.path.abspath(args.out_csv)}")
    print(f"Saved HTML: {os.path.abspath(args.html_out)}")
    if args.open_html: webbrowser.open("file://" + os.path.abspath(args.html_out))

if __name__ == "__main__":
    main()
