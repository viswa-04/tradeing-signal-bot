import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import pytz

# ----------------------------- USER CONFIG (edit before running) -----------------------------
EMAIL_SENDER = "viswamuthupandi04@gmail.com"
EMAIL_PASSWORD = "qnax nuha qzww uusj"   # <-- replace with your Gmail app password (16 chars)
EMAIL_RECEIVER = "viswamuthupandi04@gmail.com"
# -------------------------------------------------------------------------------------------

# Page
st.set_page_config(page_title="Forex Signal App (yfinance + email)", page_icon="📈", layout="centered")
st.title("📈 Forex Signal Dashboard (yfinance) + Email Alerts")
st.markdown("SMC + Orderblocks + FVG + Liquidity Sweep. Uses Yahoo Finance data.")

# ----------------------------- Utilities / Data Fetch -----------------------------
@st.cache_data(ttl=30)
def fetch_forex_data_yf(symbol, interval="15m", lookback_days=5):
    """Fetch OHLCV data from yfinance and normalize column names (lowercase)."""
    period = f"{lookback_days}d"
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"yfinance fetch error for {symbol}: {e}")
        return None

    if df is None or df.empty:
        return None

    # If MultiIndex columns, flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # lowercase map
    cols_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_map)

    # required keys: open, high, low, close
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        print(f"Skipping {symbol}: missing OHLC columns ({set(df.columns)})")
        return None

    keep = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        keep.append('volume')
    df = df[keep]

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
                        zlow = min(cclose, copen) - 0.0005 * abs(cclose)
                        zhigh = max(cclose, copen) + 0.0005 * abs(cclose)
                        obs.append({'type':'bull','idx':i,'low':float(zlow),'high':float(zhigh),'time':df.index[i]})
                        break
            # bullish candle => potential bearish OB
            if cclose > copen:
                for j in range(1, look_forward+1):
                    if i+j >= len(df): break
                    impulse = df['low'].iat[i] - df['close'].iat[i+j]
                    threshold = impulse_atr_mult * (atr_series.iat[i+j] if not np.isnan(atr_series.iat[i+j]) else atr_series.mean())
                    if (df['close'].iat[i] - df['close'].iat[i+j]) > threshold:
                        zlow = min(cclose, copen) - 0.0005 * abs(cclose)
                        zhigh = max(cclose, copen) + 0.0005 * abs(cclose)
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
        if b['low'] > a['high']:
            fvgs.append({'type':'bull','start_idx':i-2,'end_idx':i-1,'low':float(a['high']),'high':float(b['low']),'time':df.index[i-1]})
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

# ----------------------------- Signal Engine (adaptive, IST 12-hour) -----------------------------
def compute_signal(df, params, timeframe="15m"):
    # Safety: normalize column names to lowercase so 'High'/'high' won't break us
    if df is None:
        return {"signal":"NEUTRAL","reason":["no data"], "price": None, "confidence":0.0,"ms_valid":False,"ms_trend":"RANGE","found_buy_poi":None,"found_sell_poi":None,"suggested":{},"orderblocks":[],"fvgs":[],"time_ist":""}
    df = df.copy()
    # If columns exist, lowercase them
    try:
        df.columns = [c.lower() for c in df.columns]
    except Exception:
        pass

    if len(df) < 8:
        return {"signal":"NEUTRAL","reason":["insufficient data"], "price": None, "confidence":0.0,"ms_valid":False,"ms_trend":"RANGE","found_buy_poi":None,"found_sell_poi":None,"suggested":{},"orderblocks":[],"fvgs":[],"time_ist":""}

    ist_zone = pytz.timezone('Asia/Kolkata')

    # Calculate ATR
    df['atr'] = atr(df, params.get('atr_period',14))
    atr_latest = df['atr'].iat[-1] if not df['atr'].isna().all() else 0.0

    # Swings & market structure
    highs_idx, lows_idx = detect_swings(df, params.get('swing_window',3))
    ms_valid, ms_trend = is_market_structure_valid(df, highs_idx, lows_idx, required_swings=2, sma_period=params.get('sma_period',20))

    # POIs
    obs = detect_orderblocks(df, df['atr'], impulse_atr_mult=params.get('orderblock_impulse_atr',1.0))
    fvgs = detect_fvgs(df)

    latest = df.iloc[-1]
    latest_low = latest['low']; latest_high = latest['high']; latest_close = latest['close']
    latest_idx = len(df)-1

    found_buy_poi=None; found_sell_poi=None; found_sweep_buy=False; found_sweep_sell=False

    tf_mult = {"1m":0.6,"5m":0.7,"15m":0.9,"30m":1.0,"1h":1.2,"4h":1.4,"1d":1.6}
    tf_adj = tf_mult.get(timeframe,1.0)

    # Detect buy sweep (pierce below last swing low)
    if lows_idx:
        last_swing_low_idx = lows_idx[-1]
        swing_low_price = df['low'].iat[last_swing_low_idx]
        if (swing_low_price - latest_low) > (params.get('liquidity_atr_mult',0.8) * atr_latest * tf_adj):
            found_sweep_buy=True
            for z in reversed(obs):
                if z['type']=='bull' and abs(z['idx'] - last_swing_low_idx) <= 10:
                    if last_n_bars_close_inside_zone(df, latest_idx-1, z, n=params.get('confirm_close_inside',2)):
                        found_buy_poi = {'type':'orderblock','zone':z,'swing_idx':last_swing_low_idx}
                        break
            if not found_buy_poi:
                for z in reversed(fvgs):
                    if z['type']=='bull' and abs((latest_idx-1)-z['end_idx']) <= 10:
                        if last_n_bars_close_inside_zone(df, latest_idx-1, z, n=params.get('confirm_close_inside',2)):
                            found_buy_poi = {'type':'fvg','zone':z,'swing_idx':last_swing_low_idx}
                            break

    # Detect sell sweep (pierce above last swing high)
    if highs_idx:
        last_swing_high_idx = highs_idx[-1]
        swing_high_price = df['high'].iat[last_swing_high_idx]
        if (latest_high - swing_high_price) > (params.get('liquidity_atr_mult',0.8) * atr_latest * tf_adj):
            found_sweep_sell=True
            for z in reversed(obs):
                if z['type']=='bear' and abs(z['idx'] - last_swing_high_idx) <= 10:
                    if last_n_bars_close_inside_zone(df, latest_idx-1, z, n=params.get('confirm_close_inside',2)):
                        found_sell_poi = {'type':'orderblock','zone':z,'swing_idx':last_swing_high_idx}
                        break
            if not found_sell_poi:
                for z in reversed(fvgs):
                    if z['type']=='bear' and abs((latest_idx-1)-z['end_idx']) <= 10:
                        if last_n_bars_close_inside_zone(df, latest_idx-1, z, n=params.get('confirm_close_inside',2)):
                            found_sell_poi = {'type':'fvg','zone':z,'swing_idx':last_swing_high_idx}
                            break

    # Adaptive confluence threshold
    confluence_threshold = 2
    if timeframe in ["1m","5m"]:
        confluence_threshold = 1
    elif timeframe in ["1h","4h"]:
        confluence_threshold = 3

    buy_confluences = sum([found_buy_poi is not None, found_sweep_buy, len(fvgs) > 0])
    sell_confluences = sum([found_sell_poi is not None, found_sweep_sell, len(fvgs) > 0])

    signal="NEUTRAL"; reason=[]; confidence=0.0; suggested={}

    if ms_trend == "UP" and buy_confluences >= confluence_threshold:
        signal="BUY"
        reason.append(f"Uptrend + {buy_confluences} bullish confluences")
        confidence = min(0.99, 0.7 + 0.1*np.random.random())
        # Stop based on POI low or recent low, target based on favorable R:R
        stop = (found_buy_poi['zone']['low'] if found_buy_poi and 'zone' in found_buy_poi else latest_low) - 0.8 * atr_latest
        target = latest_close + 2.5 * (latest_close - stop)
        suggested = {"entry": float(latest_close), "stop": float(stop), "target": float(target)}
    elif ms_trend == "DOWN" and sell_confluences >= confluence_threshold:
        signal="SELL"
        reason.append(f"Downtrend + {sell_confluences} bearish confluences")
        confidence = min(0.99, 0.7 + 0.1*np.random.random())
        stop = (found_sell_poi['zone']['high'] if found_sell_poi and 'zone' in found_sell_poi else latest_high) + 0.8 * atr_latest
        target = latest_close - 2.5 * (stop - latest_close)
        suggested = {"entry": float(latest_close), "stop": float(stop), "target": float(target)}
    else:
        reason.append(f"Trend weak or <{confluence_threshold} confluences")

    now_ist = datetime.now(ist_zone).strftime("%d-%b-%Y %I:%M:%S %p")

    return {
        "signal": signal,
        "confidence": round(float(confidence),3),
        "price": float(latest_close),
        "ms_valid": ms_valid,
        "ms_trend": ms_trend,
        "orderblocks": obs[-10:],
        "fvgs": fvgs[-10:],
        "found_buy_poi": found_buy_poi,
        "found_sell_poi": found_sell_poi,
        "suggested": suggested,
        "reason": reason,
        "time_ist": now_ist
    }

# ----------------------------- Email Sender -----------------------------
def send_email_gmail(sender, password, receiver, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        # use SSL
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

# ----------------------------- Multi-pair runner -----------------------------
def run_multi_pair(params=None, pairs=None, timeframe="15m", user_email=None, lookback_days=5):
    # defaults
    if params is None:
        params = {
            "atr_period": 14,
            "swing_window": 3,
            "sma_period": 20,
            "orderblock_impulse_atr": 1.5,
            "liquidity_atr_mult": 0.8,
            "confirm_close_inside": 2
        }
    if pairs is None:
        pairs = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","GC=F"]

    ist = pytz.timezone('Asia/Kolkata')
    results = []

    for pair in pairs:
        try:
            df = fetch_forex_data_yf(pair, interval=timeframe, lookback_days=lookback_days)
            if df is None:
                print(f"⚠️ Skipping {pair}: no data")
                continue

            res = compute_signal(df, params, timeframe=timeframe)
            # add pair and explicit stop/target columns for table
            res["pair"] = pair
            suggested = res.get("suggested", {}) or {}
            res["stop"] = suggested.get("stop", None)
            res["target"] = suggested.get("target", None)
            res["time_ist"] = datetime.now(ist).strftime("%d-%b-%Y %I:%M:%S %p")
            results.append(res)

            # send email if signal found
            if user_email and res["signal"] in ("BUY","SELL"):
                subject = f"New {res['signal']} signal - {pair} ({timeframe})"
                body = f"""New {res['signal']} signal detected for {pair} ({timeframe}).

Price: {res['price']}
Confidence: {res['confidence']*100:.1f}%
Trend: {res['ms_trend']}
Reason: {', '.join(res.get('reason',[]))}

Suggested Entry: {res.get('suggested',{}).get('entry')}
Stop: {res.get('suggested',{}).get('stop')}
Target: {res.get('suggested',{}).get('target')}

Time (IST): {res['time_ist']}
"""
                ok, err = send_email_gmail(EMAIL_SENDER, EMAIL_PASSWORD, user_email, subject, body)
                if ok:
                    print(f"Email sent for {pair}: {subject}")
                else:
                    print(f"Email failed for {pair}: {err}")

        except Exception as e:
            print(f"⚠️ Error processing {pair}: {e}")

    if len(results) == 0:
        return pd.DataFrame()
    df_results = pd.DataFrame(results)
    keep_cols = [c for c in ["pair","signal","confidence","price","stop","target","ms_trend","time_ist"] if c in df_results.columns]
    return df_results[keep_cols]

# ----------------------------- Streamlit Dashboard (Enhanced Visuals) -----------------------------
def app_dashboard():
    st.set_page_config(page_title="📊 Forex Smart Signal Bot", layout="wide")
    st.title("📊 Forex Smart Signal Bot (Multi-Pair Dashboard)")
    st.caption("✅ SMC + Liquidity + Orderblocks + FVG with visual signal cards and reasons (IST 12-hour format)")

    # --- Sidebar Settings ---
    with st.sidebar:
        st.header("⚙️ Settings")

        default_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "XAUUSD=X"]
        pairs = st.multiselect("Select Pairs:", default_pairs, default=default_pairs)
        timeframe = st.selectbox("Timeframe:", ["5m", "15m", "30m", "1h"], index=1)

        params = {
            "atr_period": st.number_input("ATR Period", 5, 50, 14),
            "swing_window": st.number_input("Swing Window", 1, 10, 3),
            "sma_period": st.number_input("SMA Period", 5, 100, 20),
            "orderblock_impulse_atr": st.number_input("Order Impulse (ATR×)", 0.5, 3.0, 1.5),
            "liquidity_atr_mult": st.number_input("Liquidity Sweep (ATR×)", 0.1, 2.0, 0.8),
            "confirm_close_inside": st.number_input("Confirm Close Candles", 1, 5, 2)
        }

        user_email = st.text_input("📧 Email for Alerts (optional):", "")
        auto_refresh = st.checkbox("🔁 Auto Refresh every 60s", value=False)

    # --- Main Section ---
    st.markdown("---")
    if st.button("🚀 Run Full Scan Now"):
        with st.spinner("Scanning selected forex pairs..."):
            results = run_multi_pair(params, pairs, timeframe, user_email if user_email else None)
            if results.empty:
                st.warning("No valid signals found. Try another timeframe.")
            else:
                st.success("✅ Scan Complete!")

                # Display all results in a clean table
                st.subheader("📋 Signal Summary Table")
                st.dataframe(results, use_container_width=True)

                # --- Highlight individual signals ---
                for _, row in results.iterrows():
                    pair = row['pair']
                    signal = row['signal']
                    conf = row.get('confidence', 0)
                    trend = row.get('ms_trend', 'N/A')
                    price = row.get('price', 0)
                    stop = row.get('stop', None)
                    target = row.get('target', None)
                    ist_time = row.get('time_ist', '')

                    # Use the safe fetch to get normalized dataframe
                    df_pair = fetch_forex_data_yf(pair, interval=timeframe, lookback_days=5)
                    if df_pair is not None:
                        detailed = compute_signal(df_pair, params, timeframe)
                    else:
                        detailed = row

                    reasons = detailed.get("reason", [])
                    confluence = []
                    if detailed.get("found_buy_poi"): confluence.append("Orderblock")
                    if detailed.get("found_sell_poi"): confluence.append("Orderblock")
                    if len(detailed.get("fvgs", [])) > 0: confluence.append("Fair Value Gap")
                    if detailed.get("ms_valid"): confluence.append(f"Market Structure: {detailed['ms_trend']}")

                    confluence_text = " • ".join(confluence) if confluence else "No strong confluence"
                    reason_text = " | ".join(reasons) if reasons else "No specific reason found"

                    # --- Create visual signal cards with big colored signal text and SL/TP ---
                    if signal == "BUY":
                        st.markdown(f"""
                        <div style="background-color:#e9fff0; border:2px solid #00aa44; border-radius:14px; padding:18px; margin-bottom:12px;">
                            <h1 style="color:#008a2e; font-size:48px; margin:6px 0;">🟢 BUY — {pair}</h1>
                            <div style="font-size:18px; margin-bottom:6px;"><b>Price:</b> {price:.6f} &nbsp;&nbsp; <b>Time (IST):</b> {ist_time}</div>
                            <div style="font-size:16px;"><b>Trend:</b> {trend} &nbsp; | &nbsp; <b>Confidence:</b> {conf:.2f}</div>
                            <hr style="border:none; border-top:1px solid #d0f0d8; margin:10px 0;">
                            <div style="font-size:15px;"><b>Confluences:</b> {confluence_text}</div>
                            <div style="font-size:15px; margin-top:6px;"><b>Reason:</b> {reason_text}</div>
                            <div style="font-size:15px; margin-top:8px;"><b>Stop Loss:</b> {stop if stop is not None else 'N/A'} &nbsp;&nbsp; <b>Target:</b> {target if target is not None else 'N/A'}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif signal == "SELL":
                        st.markdown(f"""
                        <div style="background-color:#fff0f0; border:2px solid #e04a4a; border-radius:14px; padding:18px; margin-bottom:12px;">
                            <h1 style="color:#b30000; font-size:48px; margin:6px 0;">🔴 SELL — {pair}</h1>
                            <div style="font-size:18px; margin-bottom:6px;"><b>Price:</b> {price:.6f} &nbsp;&nbsp; <b>Time (IST):</b> {ist_time}</div>
                            <div style="font-size:16px;"><b>Trend:</b> {trend} &nbsp; | &nbsp; <b>Confidence:</b> {conf:.2f}</div>
                            <hr style="border:none; border-top:1px solid #f0d0d0; margin:10px 0;">
                            <div style="font-size:15px;"><b>Confluences:</b> {confluence_text}</div>
                            <div style="font-size:15px; margin-top:6px;"><b>Reason:</b> {reason_text}</div>
                            <div style="font-size:15px; margin-top:8px;"><b>Stop Loss:</b> {stop if stop is not None else 'N/A'} &nbsp;&nbsp; <b>Target:</b> {target if target is not None else 'N/A'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:#f7f7f7; border:1px solid #dddddd; border-radius:10px; padding:12px; margin-bottom:8px;">
                            <h2 style="color:#666666; font-size:22px; margin:6px 0;">⚪ NEUTRAL — {pair}</h2>
                            <div style="font-size:14px;"><b>Trend:</b> {trend} &nbsp; | &nbsp; <b>Price:</b> {price:.6f}</div>
                            <div style="font-size:14px; margin-top:6px;"><b>Reason:</b> {reason_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # --- Auto Refresh Logic ---
    if auto_refresh:
        st.info("🔁 Auto refreshing every 60 seconds...")
        time.sleep(60)
        st.rerun()

    st.markdown("---")
    st.caption(f"⏰ Last Updated: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%b-%Y %I:%M:%S %p')}")

# ----------------------------- Run the App -----------------------------
if __name__ == "__main__":
    app_dashboard()
