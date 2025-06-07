import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from datetime import datetime, timedelta


# --- Asset Classes Config ---
asset_classes_config = {
    "QQQ": {
        "risk_on_assets": ["SPY", "QQQ", "HYG", "XLF", "XLK"],
        "risk_off_assets": ["TLT", "GLD", "IEF", "XLU", "CHF=X"]
    },
    "SPY": {
        "risk_on_assets": ["SPY", "QQQ", "HYG", "XLF", "XLK"],
        "risk_off_assets": ["TLT", "GLD", "IEF", "XLU", "CHF=X"]
    },
    "^GDAXI": {
        "risk_on_assets": ["^GDAXI", "EWG", "XLF"],
        "risk_off_assets": ["GLD", "CHF=X"]
    },
    "^SSMI": {
        "risk_on_assets": ["^SSMI"],
        "risk_off_assets": []
    }
}

def download_asset_data(tickers, start_date, end_date):
    today = pd.to_datetime(datetime.today().date())
    end_dt = pd.to_datetime(end_date)

    # Step 1: Download daily data
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    # Step 2: If today is the end_date, check if today's data is missing
    if end_dt.date() == today.date():
        def add_intraday_latest(ticker):
            try:
                for interval in ['1m', '30m', '60m']:
                    intraday = yf.Ticker(ticker).history(period='1d', interval=interval)
                    if not intraday.empty:
                        # Take last intraday row (latest timestamp)
                        last = intraday.iloc[[-1]]
        
                        # Normalize index to date only (naive, without timezone)
                        date_only = last.index[0].date()
        
                        daily = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
        
                        # Make sure daily index is datetime.date, or convert to datetime with no timezone
                        daily_index_dates = daily.index.normalize().date if hasattr(daily.index, "normalize") else daily.index.date
        
                        # Check if that date is in daily data index
                        if pd.Timestamp(date_only) in daily.index.normalize():
                            # Replace the daily row for that date with intraday latest row
                            # But first, set last's index to match daily index type exactly
                            new_index = daily.index[daily.index.normalize() == pd.Timestamp(date_only)][0]
                            last.index = [new_index]
        
                            # Drop old row(s) for that date
                            daily = daily[~(daily.index.normalize() == pd.Timestamp(date_only))]
        
                        else:
                            # If date not present, set last.index to normalized datetime with no timezone
                            last.index = [pd.Timestamp(date_only)]
        
                        # Append intraday latest row
                        daily = pd.concat([daily, last])
                        daily = daily.sort_index()
                        return daily
        
            except Exception as e:
                print(f"Error fetching intraday data for {ticker}: {e}")
        
            return data[ticker] if isinstance(data.columns, pd.MultiIndex) else data

    return data
def compute_average_close(data, tickers):
    if not tickers:
        return pd.Series(dtype=float)
    return pd.concat([data[ticker]['Close'] for ticker in tickers], axis=1).mean(axis=1)

def compute_average_volume(data, tickers, fallback_index=None):
    if len(tickers) == 0:
        if fallback_index is not None:
            return pd.Series(0, index=fallback_index)
        raise ValueError("compute_average_volume() called with no tickers and no fallback_index.")
    return pd.concat([data[ticker]['Volume'] for ticker in tickers], axis=1).mean(axis=1)

def williams_r(series, lookback=14):
    highest_high = series.rolling(window=lookback).max()
    lowest_low = series.rolling(window=lookback).min()
    return (highest_high - series) / (highest_high - lowest_low) * -100

def generate_risk_regime(wr_series, threshold=-50):
    return pd.Series(np.where(wr_series > threshold, "Risk-On", "Risk-Off"), index=wr_series.index)

def compute_risk_analysis_for_ticker(ticker, start_date, end_date, lookback=14):
    config = asset_classes_config.get(ticker)
    if not config:
        st.error(f"Ticker {ticker} not found in asset_classes_config")
        return None

    risk_on_assets = config["risk_on_assets"]
    risk_off_assets = config["risk_off_assets"]
    all_assets = list(set(risk_on_assets + risk_off_assets))

    data = download_asset_data(all_assets, start_date, end_date)

    if "CHF=X" in data.columns.levels[0]:
        data["CHF=X", "Close"] = 1 / data["CHF=X", "Close"]

    risk_on_close = compute_average_close(data, risk_on_assets)
    risk_off_close = compute_average_close(data, risk_off_assets)
    risk_on_volume = compute_average_volume(data, risk_on_assets)
    risk_off_volume = compute_average_volume(data, risk_off_assets, fallback_index=risk_on_close.index)

    composite_index = risk_on_close - risk_off_close if not risk_off_close.empty else risk_on_close

    wr_values = williams_r(composite_index, lookback)
    regimes = generate_risk_regime(wr_values)

    return {
        "composite_index": composite_index,
        "williams_r": wr_values,
        "regimes": regimes,
        "risk_on_volume": risk_on_volume,
        "risk_off_volume": risk_off_volume,
        "data": data,
        "risk_on_assets": risk_on_assets,
        "risk_off_assets": risk_off_assets
    }

def plot_with_risk_regime(ticker, results):
    data = results['data']
    regimes = results['regimes']

    close = data[ticker]['Close'].dropna()
    aligned_regimes = regimes.loc[close.index].ffill()

    rolling_vol = close.pct_change().rolling(window=30).std() * (252 ** 0.5)

    vix = None
    if '^VIX' in data.columns.levels[0]:
        vix = data['^VIX']['Close'].reindex(close.index).ffill()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(close, label=f'{ticker} Price', color='black')
    ax1.set_ylabel('Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    current_regime = aligned_regimes.iloc[0]
    start_date = aligned_regimes.index[0]

    for i in range(1, len(aligned_regimes)):
        if aligned_regimes.iloc[i] != current_regime:
            end_date = aligned_regimes.index[i]
            ax1.axvspan(start_date, end_date,
                        color='green' if current_regime == 'Risk-On' else 'red',
                        alpha=0.2)
            current_regime = aligned_regimes.iloc[i]
            start_date = end_date

    ax1.axvspan(start_date, aligned_regimes.index[-1],
                color='green' if current_regime == 'Risk-On' else 'red',
                alpha=0.2)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Volatility / VIX', color='blue')
    ax2.plot(rolling_vol, label='30d Rolling Volatility', color='blue', linestyle='--')
    if vix is not None:
        ax2.plot(vix, label='VIX Index', color='orange')

    ax2.tick_params(axis='y', labelcolor='blue')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    ax1.set_title(f"{ticker} with Risk Regime, 30d Volatility, and VIX")

    st.pyplot(fig)

def simulate_strategies(ticker, results, start_capital=10000, offset_days=0):
    data = results['data']
    regimes = results['regimes']

    close = data[ticker]['Close'].dropna()
    aligned_regimes = regimes.reindex(close.index).ffill()

    if offset_days != 0:
        aligned_regimes = aligned_regimes.shift(offset_days)

    position = (aligned_regimes == "Risk-On").astype(int)
    returns = close.pct_change().fillna(0)
    strat_a_value = (1 + position * returns).cumprod() * start_capital
    strat_b_value = (1 + returns).cumprod() * start_capital

    trades = []
    in_trade = False
    capital = start_capital

    for i in range(1, len(close)):
        if not in_trade and position.iloc[i] == 1 and position.iloc[i - 1] == 0:
            entry_date = close.index[i]
            entry_price = close.iloc[i]
            in_trade = True
        elif in_trade and (position.iloc[i] == 0 or i == len(close) - 1):
            exit_date = close.index[i]
            exit_price = close.iloc[i]
            holding_period = (exit_date - entry_date).days
            pct_return = (exit_price - entry_price) / entry_price
            capital *= (1 + pct_return)

            trades.append({
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": exit_date,
                "Exit Price": exit_price,
                "Holding Period (Days)": holding_period,
                "Return %": pct_return * 100,
                "Capital After Trade": capital
            })
            in_trade = False

    trades_df = pd.DataFrame(trades)

    st.subheader(f"Strategy Simulation for {ticker}")
    st.line_chart(pd.DataFrame({
        "Strategy A (Regime based)": strat_a_value,
        "Strategy B (Buy & Hold)": strat_b_value
    }))

    st.markdown("### Trades Summary")
    st.dataframe(trades_df)

    # Calculate duration metrics
    risk_on_days = (aligned_regimes == "Risk-On").sum()
    risk_off_days = (aligned_regimes == "Risk-Off").sum()
    total_days = risk_on_days + risk_off_days
    total_years = total_days / 365.25

    # Identify transitions to count trades
    transitions = aligned_regimes.ne(aligned_regimes.shift())
    risk_on_entries = (aligned_regimes == "Risk-On") & transitions
    trades = risk_on_entries.sum()
    trades_per_year = trades / total_years if total_years > 0 else 0

    # Average duration in each regime
    risk_on_periods = aligned_regimes[aligned_regimes == "Risk-On"]
    risk_off_periods = aligned_regimes[aligned_regimes == "Risk-Off"]
    avg_days_risk_on = risk_on_days / trades if trades > 0 else 0
    avg_days_risk_off = risk_off_days / trades if trades > 0 else 0

    # Separate winning and losing trades
    wins = [r for r in returns if r > 0]
    losses = [abs(r) for r in returns if r <= 0]

    # Number of winning trades
    num_wins = sum(1 for r in returns if r > 0)

    # Total trades
    total_trades = len(returns)

    # Win rate as percentage
    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0

    metric_cols = st.columns(6)
    metric_cols[0].metric("Trades/Year", f"{trades_per_year:.1f}")
    metric_cols[1].metric("Risk-On Days", risk_on_days)
    metric_cols[2].metric("Risk-Off Days", risk_off_days)
    metric_cols[3].metric("Avg Days Risk-On", f"{avg_days_risk_on:.1f}")
    metric_cols[4].metric("Avg Days Risk-Off", f"{avg_days_risk_off:.1f}")
    metric_cols[5].metric("Win Rate", f"{win_rate:.1f}%")


# --- Step 6: Detailed Risk Regime Info ---
def get_detailed_risk_regime_infos(results, ticker):
    regimes = results['regimes'].ffill()
    composite = results['composite_index']
    wr = results['williams_r']
    vol_on = results['risk_on_volume']
    vol_off = results['risk_off_volume']
    data = results['data']

    if ticker not in data.columns.levels[0]:
        raise ValueError(f"{ticker} not in downloaded data.")

    price_series = data[ticker]['Close'].dropna()

    df = pd.DataFrame({
        "Regime": regimes,
        "Composite": composite,
        "Williams %R": wr,
        "Risk-On Volume": vol_on,
        "Risk-Off Volume": vol_off,
        "Price": price_series
    }).dropna()

    info = []
    current_regime = df["Regime"].iloc[0]
    start_date = df.index[0]

    for i in range(1, len(df)):
        if df["Regime"].iloc[i] != current_regime:
            end_date = df.index[i]
            period = df.loc[start_date:end_date]

            wr_start = period["Williams %R"].iloc[0]
            vol_ratio = (period["Risk-On Volume"].mean() / period["Risk-Off Volume"].mean()) if period["Risk-Off Volume"].mean() != 0 else np.nan
            price_change = (period["Price"].iloc[-1] - period["Price"].iloc[0]) / period["Price"].iloc[0]

            info.append({
                "Regime": current_regime,
                "Start Date": start_date,
                "End Date": end_date,
                "Williams %R Start": wr_start,
                "Volume Ratio (Risk-On / Risk-Off)": vol_ratio,
                "Price Change %": price_change * 100
            })

            current_regime = df["Regime"].iloc[i]
            start_date = df.index[i]

    # Add last regime period info
    period = df.loc[start_date:]
    wr_start = period["Williams %R"].iloc[0]
    vol_ratio = (period["Risk-On Volume"].mean() / period["Risk-Off Volume"].mean()) if period["Risk-Off Volume"].mean() != 0 else np.nan
    price_change = (period["Price"].iloc[-1] - period["Price"].iloc[0]) / period["Price"].iloc[0]
    info.append({
        "Regime": current_regime,
        "Start Date": start_date,
        "End Date": df.index[-1],
        "Williams %R Start": wr_start,
        "Volume Ratio (Risk-On / Risk-Off)": vol_ratio,
        "Price Change %": price_change * 100
    })

    return pd.DataFrame(info)

def get_detailed_risk_regime_infos(results, ticker):
    regimes = results['regimes'].ffill()
    composite = results['composite_index']
    wr = results['williams_r']
    vol_on = results['risk_on_volume']
    vol_off = results['risk_off_volume']
    data = results['data']

    if ticker not in data.columns.levels[0]:
        raise ValueError(f"{ticker} not in downloaded data.")

    price_series = data[ticker]['Close'].dropna()

    df = pd.DataFrame({
        "Regime": regimes,
        "Composite": composite,
        "Williams %R": wr,
        "Risk-On Volume": vol_on,
        "Risk-Off Volume": vol_off,
        "Price": price_series
    }).dropna()

    # Add this check here:
    if df.empty:
        # Return empty DataFrame so caller handles it gracefully
        return pd.DataFrame()

    info = []
    current_regime = df["Regime"].iloc[0]
    start_date = df.index[0]

    for i in range(1, len(df)):
        if df["Regime"].iloc[i] != current_regime:
            end_date = df.index[i]
            period = df.loc[start_date:end_date]

            wr_start = period["Williams %R"].iloc[0]
            vol_ratio = (period["Risk-On Volume"].mean() / period["Risk-Off Volume"].mean()) if period["Risk-Off Volume"].mean() != 0 else np.nan
            price_change = (period["Price"].iloc[-1] - period["Price"].iloc[0]) / period["Price"].iloc[0]

            info.append({
                "Regime": current_regime,
                "Start Date": start_date,
                "End Date": end_date,
                "Williams %R Start": wr_start,
                "Volume Ratio (Risk-On / Risk-Off)": vol_ratio,
                "Price Change %": price_change * 100
            })

            current_regime = df["Regime"].iloc[i]
            start_date = df.index[i]

    # Add last regime period info
    period = df.loc[start_date:]
    wr_start = period["Williams %R"].iloc[0]
    vol_ratio = (period["Risk-On Volume"].mean() / period["Risk-Off Volume"].mean()) if period["Risk-Off Volume"].mean() != 0 else np.nan
    price_change = (period["Price"].iloc[-1] - period["Price"].iloc[0]) / period["Price"].iloc[0]
    info.append({
        "Regime": current_regime,
        "Start Date": start_date,
        "End Date": df.index[-1],
        "Williams %R Start": wr_start,
        "Volume Ratio (Risk-On / Risk-Off)": vol_ratio,
        "Price Change %": price_change * 100
    })

    return pd.DataFrame(info)

def plot_technical_indicators(ticker, results):
    data = results['data']
    regimes = results['regimes']

    if ticker not in data.columns.levels[0]:
        raise ValueError(f"{ticker} not in downloaded data.")

    df = data[ticker].dropna().copy()
    df.index = pd.to_datetime(df.index)

    # Calculate Bollinger Bands
    window = 20
    df['SMA20'] = df['Close'].rolling(window).mean()
    df['STD20'] = df['Close'].rolling(window).std()
    df['BB_upper'] = df['SMA20'] + 2 * df['STD20']
    df['BB_lower'] = df['SMA20'] - 2 * df['STD20']

    # Calculate 200 EMA
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    aligned_regimes = regimes.loc[df.index].ffill()

    # Prepare colors for risk regimes
    regime_colors = {'Risk-On': 'green', 'Risk-Off': 'red'}

    # Plotting
    fig, (ax_price, ax_macd) = plt.subplots(2, 1, figsize=(14,10), sharex=True,
                                            gridspec_kw={'height_ratios': [3,1]})

    # --- Plot candlesticks ---
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values
    dates = mdates.date2num(df.index.to_pydatetime())

    width = 0.6
    width2 = 0.1
    col_up = 'green'
    col_down = 'red'

    for i in range(len(df)):
        color = col_up if c[i] >= o[i] else col_down
        ax_price.plot([dates[i], dates[i]], [l[i], h[i]], color=color)  # high-low line
        ax_price.add_patch(plt.Rectangle((dates[i]-width/2, min(o[i], c[i])),
                                         width, abs(c[i]-o[i]),
                                         facecolor=color, edgecolor=color))

    # --- Plot Bollinger Bands ---
    ax_price.plot(df.index, df['BB_upper'], color='blue', label='BB Upper')
    ax_price.plot(df.index, df['BB_lower'], color='blue', label='BB Lower')

    # --- Plot 200 EMA as dotted ---
    ax_price.plot(df.index, df['EMA200'], color='black', linestyle='dotted', label='200 EMA')

    ax_price.set_ylabel('Price')
    ax_price.legend(loc='upper left')

    # --- Plot MACD ---
    ax_macd.plot(df.index, df['MACD'], color='fuchsia', label='MACD')
    ax_macd.plot(df.index, df['MACD_signal'], color='blue', label='Signal')
    ax_macd.bar(df.index, df['MACD_hist'], color='gray', alpha=0.5, label='Histogram')

    ax_macd.set_ylabel('MACD')
    ax_macd.legend(loc='upper left')

    # --- Risk regime background shading ---
    current_regime = aligned_regimes.iloc[0]
    start_idx = df.index[0]

    for i in range(1, len(aligned_regimes)):
        if aligned_regimes.iloc[i] != current_regime:
            end_idx = df.index[i]
            ax_price.axvspan(start_idx, end_idx, color=regime_colors[current_regime], alpha=0.2)
            current_regime = aligned_regimes.iloc[i]
            start_idx = end_idx
    # Last span
    ax_price.axvspan(start_idx, df.index[-1], color=regime_colors[current_regime], alpha=0.2)

    # Formatting x-axis
    ax_macd.xaxis.set_major_locator(mdates.MonthLocator())
    ax_macd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.suptitle(f'{ticker} Price with Bollinger Bands, 200 EMA and MACD\nwith Risk Regime Background')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    st.pyplot(fig)

def get_current_risk_regime_info(ticker, results): #actually not in use
    """
    Returns info about the current risk regime for the given ticker:
    - start date of the current regime
    - last available price of the ticker
    - confidence (strength) of the regime in percentage
    
    Confidence here is derived from Williams %R value:
    - For Risk-On: closer to 0 means stronger (scale accordingly)
    - For Risk-Off: closer to -100 means stronger (scale accordingly)
    
    Adjust confidence calculation as needed.
    """
    regimes = results['regimes'].ffill()
    wr = results['williams_r']
    data = results['data']

    if ticker not in data.columns.levels[0]:
        raise ValueError(f"{ticker} not in downloaded data.")

    price_series = data[ticker]['Close'].dropna()
    regime_series = regimes.reindex(price_series.index).ffill()
    wr_series = wr.reindex(price_series.index).ffill()

    # Get current regime and its start date
    current_regime = regime_series.iloc[-1]
    # Find when this current regime started (the last transition date)
    regime_changes = regime_series != regime_series.shift()
    regime_start_date = regime_series[regime_changes].index[-1]

    # Last price
    last_price = price_series.iloc[-1]

    # Calculate confidence (example):
    # Williams %R ranges from -100 to 0
    # For Risk-On: closer to 0 is stronger => confidence = 100 + wr_value (wr_value is negative)
    # For Risk-Off: closer to -100 is stronger => confidence = abs(wr_value)
    wr_current = wr_series.iloc[-1]

    if current_regime == "Risk-On":
        confidence = 100 + wr_current  # wr_current is negative, so add to 100
    else:
        confidence = abs(wr_current)

    # Clamp confidence between 0 and 100
    confidence = max(0, min(100, confidence))

    return {
        "current_regime": current_regime,
        "regime_start_date": regime_start_date.strftime('%Y-%m-%d'),
        "last_price": last_price,
        "confidence_percent": confidence
    }


def select_random_period_fixed_range(start_year=2010):
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp.today()
    start_u = start_date.value
    end_u = end_date.value
    rand_u = np.random.randint(start_u, end_u, size=2)
    rand_dates = pd.to_datetime(rand_u)
    start_rand, end_rand = sorted(rand_dates)
    return start_rand.date(), end_rand.date()

def randomize_dates():
    start, end = select_random_period_fixed_range()
    st.session_state.start_date = start
    st.session_state.end_date = end

def main():
    # Initialize session state values if not present
    if 'start_date' not in st.session_state:
        st.session_state.start_date = pd.Timestamp.today() - pd.Timedelta(days=60)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = pd.Timestamp.today()

    st.set_page_config(layout="wide")
    st.title("Risk Regime Analysis")

    # Default dates: 1 month ago to today
    today = datetime.today().date()
    one_month_ago = today - timedelta(days=30)


    # Tickers input in 3 columns
    cols_input = st.columns(3)
    tickers = []
    default_tickers = ["QQQ", "^GDAXI", "^SSMI"]

    for i, col in enumerate(cols_input):
        tick = col.text_input(f"Ticker {i+1}", value=default_tickers[i]).upper()
        tickers.append(tick)

    # Button to randomize dates
    if st.button("Select Random Time Period"):
        randomize_dates()

    # Display inputs side-by-side, using session state for default values
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start Date", value=st.session_state.start_date)
    with col_end:
        end_date = st.date_input("End Date", value=st.session_state.end_date)

    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    #st.write(f"Selected period: **{st.session_state.start_date}** to **{st.session_state.end_date}**")

    if st.button("Run Analysis for All"):
        results_all = {}
        for ticker in tickers:
            try:
                with st.spinner(f"Processing {ticker}..."):
                    res = compute_risk_analysis_for_ticker(ticker, str(start_date), str(end_date))
                    if res is None:
                        st.error(f"Skipping {ticker} due to missing config or data.")
                        continue
                    results_all[ticker] = res
            except Exception as e:
                st.error(f"Error processing {ticker}: {e}")

        # Display results in 3 columns for the 3 tickers
        cols = st.columns(3)
        for i, ticker in enumerate(tickers):
            with cols[i]:
                st.subheader(ticker)

                res = results_all.get(ticker)
                if not res:
                    st.write("No results available.")
                    continue

                last_regime,regime_start_date,last_price,confidence_percent=get_current_risk_regime_info(ticker,res)
                # Current regime status (last regime)
                last_regime = res['regimes'].iloc[-1]
                color = "green" if last_regime == "Risk-On" else "red"
                st.markdown(f"**Current Regime:** <span style='color:{color}; font-weight:bold'>{last_regime}</span>", unsafe_allow_html=True)

                # Plot risk regime
                plot_with_risk_regime(ticker, res)
                

                # Display detailed regime info
                st.subheader("Detailed Risk Regime Information")
                detailed_df = get_detailed_risk_regime_infos(res, ticker)

                # Optional: format table (e.g., round values)
                st.dataframe(detailed_df.round(2).iloc[::-1])

                # Plot strategy comparison
                simulate_strategies(ticker, res)


                st.markdown("### Technical Indicators with Risk Regimes")
                plot_technical_indicators(ticker,res)













if __name__ == "__main__":
    main()
