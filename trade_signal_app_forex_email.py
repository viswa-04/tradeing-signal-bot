"""
trade_signal_app_forex_email.py

Streamlit Forex Signal Dashboard using Yahoo Finance (yfinance)
- Market structure validation (HH/HL / LH/LL + SMA slope)
- POI: Orderblocks + Fair Value Gaps (FVG)
- Liquidity sweep detection (wick pierce by ATR * multiplier) + reclaim
- Email notifications on new BUY/SELL signals

Author: ChatGPT (for Viswa)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from dateutil import tz

# ----------------------------- USER CONFIG -----------------------------
EMAIL_SENDER = "viswamuthupandi04@gmail.com"
EMAIL_PASSWORD = "qnax nuha qzww uusj"   # Gmail app password (16 chars)
EMAIL_RECEIVER = "viswamuthupandi04@gmail.com"

DEFAULT_PAIR = "EURUSD=X"
DEFAULT_INTERVAL = "5m"
DEFAULT_LOOKBACK_DAYS = 5
# -----------------------------------------------------------------------

st.set_page_config(page_title="Forex Signal App (yfinance + email)", page_icon="📈", layout="centered")

st.title("📈 Forex Signal Dashboard (yfinance) + Email Alerts")
st.markdown("SMC + Orderblocks + FVG + Liquidity Sweep. Uses Yahoo Finance data.")

# ----------------------------- Utilities / Data Fetch -----------------------------
@st.cache_data(ttl=30)
def fetch_forex_data(symbol=DEFAULT_PAIR, interval=DEFAULT_INTERVAL, lookback_days=DEFAULT_LOOKBACK_DAYS):
    period = f"{lookback_days}d"
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, threads=False)
    except Exception as e:
        st.error(f"yfinance fetch error: {e}")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
    df = df[['open','high','low','close','volume']]
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

# ----------------------------- Technical helpers -----------------------------
def atr(df, n=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def is_swing_high(df, idx, w):
    if idx < w or idx + w >= len(df): 
        return False
    center = df['high'].iat[idx]
    left = df['high'].iloc[idx-w:idx]
    right = df['high'].iloc[idx+1:idx+1+w]
    return (center > left.max()) and (center > right.max())

def is_swing_low(df, idx, w):
    if idx < w or idx + w >= len(df): 
        return False
    center = df['low'].iat[idx]
    left = df['low'].iloc[idx-w:idx]
    right = df['low'].iloc[idx+1:idx+1+w]
    return (center < left.min()) and (center < right.min())

def detect_swings(df, w):
    highs_idx, lows_idx = [], []
    for i in range(len(df)):
        try:
            if is_swing_high(df, i, w): highs_idx.append(i)
            if is_swing_low(df, i, w): lows_idx.append(i)
        except Exception:
            continue
    return highs_idx, lows_idx

def detect_orderblocks(df, atr_series, impulse_atr_mult=1.0, look_forward=5):
    obs = []
    for i in range(len(df) - look_forward - 1):
        try:
            copen = df['open'].iat[i]
            cclose = df['close'].iat[i]
            # bearish candle => potential bullish OB
            if cclose < copen:
                for j in range(1, look_forward+1):
                    if i+j >= len(df): break
                    impulse = df['close'].iat[i+j] - df['high'].iat[i]
                    threshold = impulse_atr_mult * (atr_series.iat[i+j] if not np.isnan(atr_series.iat[i+j]) else atr_series.mean())
                    if impulse > threshold:
                        zlow = min(cclose, copen) - 0.0005 * cclose
                        zhigh = max(cclose, copen) + 0.0005 * cclose
                        obs.append({'type':'bull','idx':i,'low':float(zlow),'high':float(zhigh),'time':df.index[i]})
                        break
            # bullish candle => potential bearish OB
            if cclose > copen:
                for j in range(1, look_forward+1):
                    if i+j >= len(df): break
                    impulse = df['low'].iat[i] - df['close'].iat[i+j]
                    threshold = impulse_atr_mult * (atr_series.iat[i+j] if not np.isnan(atr_series.iat[i+j]) else atr_series.mean())
                    if (df['close'].iat[i] - df['close'].iat[i+j]) > threshold:
                        zlow = min(cclose, copen) - 0.0005 * cclose
                        zhigh = max(cclose, copen) + 0.0005 * cclose
                        obs.append({'type':'bear','idx':i,'low':float(zlow),'high':float(zhigh),'time':df.index[i]})
                        break
        except Exception:
            continue
    return obs

def detect_fvgs(df):
    fvgs = []
    for i in range(2, len(df)):
        a = df.iloc[i-2]
        b = df.iloc[i-1]
        c = df.iloc[i]
        # bullish FVG
        if b['low'] > a['high']:
            fvgs.append({'type':'bull','start_idx':i-2,'end_idx':i-1,'low':float(a['high']),'high':float(b['low']),'time':df.index[i-1]})
        # bearish FVG
        if b['high'] < a['low']:
            fvgs.append({'type':'bear','start_idx':i-2,'end_idx':i-1,'low':float(b['high']),'high':float(a['low']),'time':df.index[i-1]})
    return fvgs

def last_n_bars_close_inside_zone(df, start_idx, zone, n=2):
    for i in range(1, n+1):
        idx = start_idx + i
        if idx >= len(df): 
            return False
        close = df['close'].iat[idx]
        if not (zone['low'] <= close <= zone['high']):
            return False
    return True

def is_market_structure_valid(df, highs_idx, lows_idx, required_swings=3, sma_period=50):
    trend = "RANGE"
    valid = False
    if len(highs_idx) >= 2 and len(lows_idx) >= 2 and min(len(highs_idx), len(lows_idx)) >= required_swings-1:
        last_highs = [df['high'].iat[i] for i in highs_idx[-2:]]
        last_lows = [df['low'].iat[i] for i in lows_idx[-2:]]
        sma = df['close'].rolling(sma_period, min_periods=1).mean()
        sma_slope = sma.iat[-1] - sma.iat[-3] if len(sma) >= 3 else 0.0
        if last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2] and sma_slope > 0:
            trend = "UP"; valid = True
        elif last_highs[-1] < last_highs[-2] and last_lows[-1] < last_lows[-2] and sma_slope < 0:
            trend = "DOWN"; valid = True
    return valid, trend

# ----------------------------- Signal Engine -----------------------------
def compute_signal(df, params):
    if df is None or len(df) < 60:
        return {"signal":"NEUTRAL","reason":["insufficient data"], "price": None}

    df = df.copy()
    df['atr'] = atr(df, params['atr_period'])
    atr_latest = df['atr'].iat[-1] if not df['atr'].isna().all() else 0.0

    highs_idx, lows_idx = detect_swings(df, params['swing_window'])
    ms_valid, ms_trend = is_market_structure_valid(df, highs_idx, lows_idx, required_swings=3, sma_period=params['sma_period'])

    obs = detect_orderblocks(df, df['atr'], impulse_atr_mult=params['orderblock_impulse_atr'])
    fvgs = detect_fvgs(df)

    latest_idx = len(df)-1
    latest = df.iloc[-1]
    latest_low = latest['low']
    latest_high = latest['high']
    latest_close = latest['close']

    found_buy_poi = None
    found_sell_poi = None

    # Detect sweeps
    if lows_idx:
        last_swing_low_idx = lows_idx[-1]
        swing_low_price = df['low'].iat[last_swing_low_idx]
        if (swing_low_price - latest_low) > (params['liquidity_atr_mult'] * atr_latest):
            for z in reversed(obs):
                if z['type']=='bull' and abs(z['idx'] - last_swing_low_idx) <= 6:
                    if last_n_bars_close_inside_zone(df, latest_idx-1, z, n=params['confirm_close_inside']):
                        found_buy_poi = {'type':'orderblock','zone':z,'swing_idx':last_swing_low_idx}
                        break

    if highs_idx:
        last_swing_high_idx = highs_idx[-1]
        swing_high_price = df['high'].iat[last_swing_high_idx]
        if (latest_high - swing_high_price) > (params['liquidity_atr_mult'] * atr_latest):
            for z in reversed(obs):
                if z['type']=='bear' and abs(z['idx'] - last_swing_high_idx) <= 6:
                    if last_n_bars_close_inside_zone(df, latest_idx-1, z, n=params['confirm_close_inside']):
                        found_sell_poi = {'type':'orderblock','zone':z,'swing_idx':last_swing_high_idx}
                        break

    signal = "NEUTRAL"
    reason = []
    confidence = 0.0
    suggested = {}

    if ms_valid and ms_trend == "UP" and found_buy_poi:
        zone = found_buy_poi['zone']
        low = zone['low']; high = zone['high']
        signal = "BUY"
        reason.append("MS valid (UP) + POI + liquidity sweep reclaim")
        conf = min(0.99, 0.18 + (0.28))  # simplified confidence
        confidence = float(conf)
        stop = low - 0.5 * atr_latest
        target = latest_close + 2.0 * (latest_close - stop)
        suggested = {"entry":latest_close,"stop":float(stop),"target":float(target)}
    elif ms_valid and ms_trend == "DOWN" and found_sell_poi:
        zone = found_sell_poi['zone']
        low = zone['low']; high = zone['high']
        signal = "SELL"
        reason.append("MS valid (DOWN) + POI + liquidity sweep reclaim")
        conf = min(0.99, 0.18 + (0.28))
        confidence = float(conf)
        stop = high + 0.5 * atr_latest
        target = latest_close - 2.0 * (stop - latest_close)
        suggested = {"entry":latest_close,"stop":float(stop),"target":float(target)}
    else:
        if not ms_valid:
            reason.append("Market structure not valid")
        if not (found_buy_poi or found_sell_poi):
            reason.append("No POI + sweep confluence")

    return {
        "signal": signal,
        "confidence": round(confidence,3),
        "price": float(latest_close),
        "ms_valid": ms_valid,
        "ms_trend": ms_trend,
        "orderblocks": obs[-10:],
        "fvgs": fvgs[-10:],
        "found_buy_poi": found_buy_poi,
        "found_sell_poi": found_sell_poi,
        "suggested": suggested,
        "reason": reason
    }

# ----------------------------- Email Sender -----------------------------
def send_email_gmail(sender, password, receiver, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------------------- Streamlit UI -----------------------------
st.sidebar.header("Settings")
pair = st.sidebar.selectbox("Forex pair (Yahoo format)", ["EURUSD=X","GBPUSD=X","USDJPY=X","GBPJPY=X","XAUUSD=X","AUDUSD=X","USDCAD=X"], index=0)
interval = st.sidebar.selectbox("Interval", ["1m","2m","5m","15m","30m","60m","90m","1d"], index=2)
lookback_days = st.sidebar.slider("Lookback days", 1, 30, 5)
sw_window = st.sidebar.slider("Swing window", 2, 8, 5)
atr_period = st.sidebar.slider("ATR period", 7, 21, 14)
liquidity_mult = st.sidebar.slider("Sweep ATR multiplier x100", 10, 100, 35)/100.0
orderblock_impulse = st.sidebar.slider("Orderblock impulse (ATR x100)", 50, 300, 100)/100.0
sma_period = st.sidebar.slider("SMA period", 20, 100, 50)
confirm_inside = st.sidebar.slider("Confirm reclaim bars", 1, 3, 2)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
refresh_interval = st.sidebar.slider("Auto-refresh interval sec", 5, 120, 15)
email_alerts = st.sidebar.checkbox("Enable email alerts", value=False)

st.markdown(f"**Pair:** {pair}  •  **Interval:** {interval}  •  **Lookback:** {lookback_days} days")

params = {
    'swing_window': sw_window,
    'atr_period': atr_period,
    'liquidity_atr_mult': liquidity_mult,
    'orderblock_impulse_atr': orderblock_impulse,
    'sma_period': sma_period,
    'confirm_close_inside': confirm_inside
}

if 'history' not in st.session_state: st.session_state['history'] = []
if 'last_signal' not in st.session_state: st.session_state['last_signal'] = None
if 'last_signal_time' not in st.session_state: st.session_state['last_signal_time'] = None

placeholder = st.empty()

def run_once_and_display():
    df = fetch_forex_data(pair, interval, lookback_days)
    if df is None or df.empty:
        st.error("No data returned from yfinance.")
        return
    res = compute_signal(df, params)

    # display
    col1, col2 = st.columns([2,1])
    with col1:
        sig = res['signal']
        color = "black"
        if sig == "BUY": color="green"
        elif sig=="SELL": color="red"
        st.markdown(f"<h1 style='text-align:center;color:{color};font-size:64px'>{sig}</h1>", unsafe_allow_html=True)
        if res['price'] is not None:
            st.write(f"Price: {res['price']:.6f} | Confidence: {res['confidence']*100:.1f}% | Trend: {res['ms_trend']}")
        st.write("Reason:", ", ".join(res.get('reason',[])))
        if res.get('suggested'):
            s = res['suggested']
            st.write("Suggested Entry / Stop / Target:")
            st.write(f"Entry: {s['entry']:.6f}   Stop: {s['stop']:.6f}   Target: {s['target']:.6f}")

    with col2:
        st.subheader("POI (recent)")
        st.write("Orderblocks (last):")
        for z in res['orderblocks'][::-1]:
            st.write(f"{z['type'].upper()} @ {z['time']}  zone:{z['low']:.6f}-{z['high']:.6f}")
        st.write("---")
        st.write("FVGs (last):")
        for z in res['fvgs'][::-1]:
            st.write(f"{z['type'].upper()} @ {z['time']}  zone:{z['low']:.6f}-{z['high']:.6f}")

    st.subheader("Latest candles (tail)")
    st.dataframe(df[['open','high','low','close','volume']].tail(12))

    # save history
    now_local = datetime.now(timezone.utc).astimezone(tz.tzlocal()).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['history'].append({"time":now_local,"signal":res['signal'],"confidence":res['confidence'],"price":res['price']})
    st.subheader("Recent signals")
    st.dataframe(pd.DataFrame(st.session_state['history']).tail(40), use_container_width=True)

    # Email alerts
    if email_alerts and res['signal'] in ("BUY","SELL"):
        last = st.session_state.get('last_signal', None)
        if last != res['signal']:
            subject = f"New {res['signal']} signal - {pair} ({interval})"
            body = f"""New {res['signal']} signal detected for {pair} ({interval}).

    Price: {res.get('price', 'N/A')}
    Confidence: {res.get('confidence', 0)*100:.1f}%
    Trend: {res.get('ms_trend', 'N/A')}
    Reason: {', '.join(res.get('reason', []))}

    Suggested Entry: {res.get('suggested', {}).get('entry', 'N/A')}
    Stop: {res.get('suggested', {}).get('stop', 'N/A')}
    Target: {res.get('suggested', {}).get('target', 'N/A')}

    Time: {now_local}
    """

            ok, err = send_email_gmail(EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, subject, body)
            if ok:
                st.success(f"Email sent: {subject}")
                st.session_state['last_signal'] = res['signal']
                st.session_state['last_signal_time'] = now_local
            else:
                st.error(f"Email failed: {err}")

# ----------------------------- Refresh / Auto-refresh -----------------------------
refresh_col1, refresh_col2 = st.columns([1,1])
with refresh_col1:
    if st.button("Refresh Signal"):
        run_once_and_display()

if auto_refresh:
    with refresh_col2:
        st.write(f"Auto-refresh every {refresh_interval} sec")
        time.sleep(refresh_interval)
        run_once_and_display()
else:
    run_once_and_display()
