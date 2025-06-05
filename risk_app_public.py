import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Your asset config and analysis functions from before (same as above) ---

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
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    return data

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta

def download_asset_data(tickers, start_date, end_date, interval='1d'):
    """
    Downloads asset data from yfinance with robust handling of daily and intraday data,
    including appending last intraday point to daily data if available.

    Parameters:
    - tickers: str or list of str, ticker symbols.
    - start_date: str or datetime/date, start date for data.
    - end_date: str or datetime/date, end date for data.
    - interval: str, data interval like '1d', '1m', '5m', etc.

    Returns:
    - pandas.DataFrame with downloaded data.
    """
    max_days_lookup = {
        '1m': 7,
        '2m': 30,
        '5m': 60,
        '15m': 60,
        '30m': 60,
        '60m': 60,
        '1h': 60,
        '1d': 3650
    }

    # Convert string dates to datetime.date if needed
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()

    # Enforce max days limit per interval
    max_days = max_days_lookup.get(interval, 60)
    if (end_date - start_date).days > max_days:
        start_date = end_date - timedelta(days=max_days)
        st.warning(f"Date range trimmed to last {max_days} days due to interval limit.")

    st.info(f"Fetching data for {tickers} from {start_date} to {end_date} with interval '{interval}'")

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date + timedelta(days=1),  # yfinance end is exclusive
            interval=interval,
            group_by='ticker' if isinstance(tickers, list) else False,
            auto_adjust=True,
            progress=False
        )

        if data is None or data.empty:
            st.warning("No data downloaded.")
            return pd.DataFrame()

        st.write(f"Downloaded {tickers} data shape: {data.shape}")

        if interval == '1d':
            today = date.today()

            # Extract latest date from data for daily interval
            if isinstance(tickers, list):
                first_ticker = tickers[0]
                if first_ticker in data.columns.levels[0]:
                    latest_date = data[first_ticker].dropna().index.max().date()
                else:
                    latest_date = data.dropna().index.max().date()
            else:
                latest_date = data.dropna().index.max().date()

            st.write(f"Latest data date: {latest_date}, Today: {today}")

            # Append last intraday data point if daily data is outdated
            if latest_date and latest_date < today:
                intraday_data = yf.download(
                    tickers,
                    period='1d',
                    interval='1m',
                    progress=False,
                    auto_adjust=True
                )
                if not intraday_data.empty:
                    st.write(f"Intraday data shape: {intraday_data.shape}")

                    last_row = intraday_data.iloc[[-1]]

                    # Append if last intraday timestamp not in daily data
                    if not last_row.index.isin(data.index).any():
                        st.write(f"Appending last intraday data point with timestamp {last_row.index[0]}")
                        data = pd.concat([data, last_row])
                else:
                    st.warning("No intraday data available to append.")
        else:
            st.write("Interval is not daily, returning downloaded data.")

        return data

    except Exception as e:
        st.warning(f"Failed to download data due to error: {e} Falling back to daily data.")
        if interval != '1d':
            return download_asset_data(tickers, start_date, end_date, interval='1d')
        else:
            st.error("Failed to fetch even daily data.")
            return pd.DataFrame()


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

def compute_risk_analysis_for_ticker(ticker, start_date="2020-01-01", end_date="2023-01-01", lookback=14):
    config = asset_classes_config.get(ticker)
    if not config:
        raise ValueError(f"Ticker {ticker} not found in asset_classes_config")

    risk_on_assets = config["risk_on_assets"]
    risk_off_assets = config["risk_off_assets"]
    all_assets = list(set(risk_on_assets + risk_off_assets))

    data = download_asset_data(all_assets, start_date, end_date,interval=selected_interval)

    # Invert CHF to match expected "CHF/USD" performance as risk-off
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

    if ticker not in data.columns.levels[0]:
        st.error(f"{ticker} not in downloaded data.")
        return

    close = data[ticker]['Close'].dropna()
    aligned_regimes = regimes.loc[close.index].ffill()

    rolling_vol = close.pct_change().rolling(window=30).std() * (252 ** 0.5)  # Annualised

    if '^VIX' in data.columns.levels[0]:
        vix = data['^VIX']['Close'].reindex(close.index).ffill()
    else:
        vix = None

    fig, ax1 = plt.subplots(figsize=(6, 4))

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
        ax2.plot(vix, label='VIX Index', color='orange', linestyle='-')

    ax2.tick_params(axis='y', labelcolor='blue')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    ax1.set_title(f"{ticker} Price & Risk Regime")
    st.pyplot(fig)

def simulate_strategies(ticker, results, start_capital=10000, offset_days=0, plot=True):
    data = results['data']
    regimes = results['regimes']

    if ticker not in data.columns.levels[0]:
        st.error(f"{ticker} not in downloaded data.")
        return None, None, None

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
    entry_date = None
    entry_price = None
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

    trade_log_df = pd.DataFrame(trades)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(strat_a_value, label="Risk-On Strategy", color='green')
        ax.plot(strat_b_value, label="Buy and Hold", color='blue')
        ax.set_title(f"Strategy Comparison ({ticker})\nOffset: {offset_days} days")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    return strat_a_value, strat_b_value, trade_log_df

def get_detailed_risk_regime_infos(results, ticker):
    regimes = results['regimes'].ffill()
    composite = results['composite_index']
    wr = results['williams_r']
    vol_on = results['risk_on_volume']
    vol_off = results['risk_off_volume']
    data = results['data']

    if ticker not in data.columns.levels[0]:
        st.error(f"{ticker} not in downloaded data.")
        return pd.DataFrame()

    close = data[ticker]['Close'].dropna()
    regimes = regimes.reindex(close.index).ffill()
    composite = composite.reindex(close.index).ffill()
    wr = wr.reindex(close.index).ffill()
    vol_on = vol_on.reindex(close.index).fillna(0)
    vol_off = vol_off.reindex(close.index).fillna(0)

    regime_changes = regimes.ne(regimes.shift())
    periods = regimes[regime_changes].index.tolist() + [regimes.index[-1]]

    detailed_records = []
    for i in range(len(periods) - 1):
        start = periods[i]
        end = periods[i + 1]
        mask = (regimes.index >= start) & (regimes.index < end)

        detailed_records.append({
            "Start": start,
            "End": end,
            "Duration (days)": (end - start).days,
            "Regime": regimes.loc[start],
            "Avg Composite Index": composite.loc[mask].mean(),
            "Avg Williams %R": wr.loc[mask].mean(),
            "Avg Risk-On Volume": vol_on.loc[mask].mean(),
            "Avg Risk-Off Volume": vol_off.loc[mask].mean(),
        })

    detailed_df = pd.DataFrame(detailed_records)
    return detailed_df

# --- Streamlit UI ---

# --- Your previous functions remain unchanged ---

# Enable wide mode by default
st.set_page_config(layout="wide")

st.title("Risk Regime Analysis")


# Interval selection
interval_options = {
    "1 Minute": "1m",
    "2 Minutes": "2m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "1 Day (recommended)": "1d",
    "1 Week": "1wk",
    "1 Month": "1mo"
}

selected_interval_label = st.selectbox("Select data interval", list(interval_options.keys()), index=6)  # default: '1 Hour'
selected_interval = interval_options[selected_interval_label]


# Default dates: 1 year ago to today
today = datetime.today().date()
one_month_ago = today - timedelta(days=30)

# Display start and end date on the same line
col_start, col_end = st.columns(2)
with col_start:
    start_date = st.date_input("Start Date", value=one_month_ago)
with col_end:
    end_date = st.date_input("End Date", value=today)


lookback = st.slider("Williams %R Lookback Period", min_value=5, max_value=50, value=14)

# Tickers input
cols_input = st.columns(3)
tickers = []
default_tickers = ["QQQ", "^GDAXI", "^SSMI"]
for i, col in enumerate(cols_input):
    tick = col.text_input(f"Ticker {i+1}", value=default_tickers[i]).upper()
    tickers.append(tick)

offset_days = st.slider("Strategy Regime Offset (days)", min_value=0, max_value=10, value=0)

if st.button("Run Analysis for All"):
    results_all = {}
    for ticker in tickers:
        try:
            with st.spinner(f"Processing {ticker}..."):
                res = compute_risk_analysis_for_ticker(ticker, str(start_date), str(end_date), lookback)
                results_all[ticker] = res
        except Exception as e:
            st.error(f"Error processing {ticker}: {e}")

    cols = st.columns(3)
    for i, ticker in enumerate(tickers):
        with cols[i]:
            st.header(ticker)


            # Get regimes Series
            regimes = results_all[ticker]['regimes'].ffill()

            # Get the current regime and its start date
            last_regime = regimes.iloc[-1]
            regime_change_idx = regimes[::-1].ne(last_regime).idxmax()
            
            #regime_start_date = regimes.index[regimes.index.get_loc(regime_change_idx) + 1] if regime_change_idx != regimes.index[0] else regimes.index[0]
            regime_idx = regimes.index.get_loc(regime_change_idx)
            if regime_idx + 1 < len(regimes.index):
                regime_start_date = regimes.index[regime_idx + 1]
            else:
                regime_start_date = regimes.index[regime_idx]  # fallback to last valid index

            # Display current regime info
            st.markdown(f"**🟢 Current Regime:** `{last_regime}` since **{regime_start_date.strftime('%Y-%m-%d')}**")

            # Initialize trade_log_df for each ticker
            trade_log_df = _,_,trades_df = simulate_strategies(ticker, results_all[ticker], offset_days=offset_days,plot=False)
            # Extract necessary series
            regimes = results_all[ticker]['regimes'].ffill()


            # Total period
            total_days = (regimes.index[-1] - regimes.index[0]).days
            total_years = total_days / 365.25

            # Trades per Year
            trades_per_year = len(trades_df) / total_years if total_years > 0 else 0

            # 2. Risk-On and 3. Risk-Off Days
            risk_on_days = (regimes == "Risk-On").sum()
            risk_off_days = (regimes == "Risk-Off").sum()

            # 4. Avg Days Risk-On and 5. Avg Days Risk-Off
            regime_changes = regimes.ne(regimes.shift()).cumsum()
            durations = regimes.groupby(regime_changes).agg(['first', 'size'])
            avg_days_risk_on = durations[durations['first'] == 'Risk-On']['size'].mean()
            avg_days_risk_off = durations[durations['first'] == 'Risk-Off']['size'].mean()

            # Show 6 metrics
            metric_cols = st.columns(6)
            metric_cols[0].metric("Trades/Year", f"{trades_per_year:.1f}")
            metric_cols[1].metric("Risk-On Days", risk_on_days)
            metric_cols[2].metric("Risk-Off Days", risk_off_days)
            metric_cols[3].metric("Avg Days Risk-On", f"{avg_days_risk_on:.1f}")
            metric_cols[4].metric("Avg Days Risk-Off", f"{avg_days_risk_off:.1f}")
            metric_cols[5].metric("Total Period (Years)", f"{total_years:.2f}")

            if ticker in results_all:
                plot_with_risk_regime(ticker, results_all[ticker])
                strat_a, strat_b, trades_df = simulate_strategies(ticker, results_all[ticker], offset_days=offset_days,plot=False)
                if trades_df is not None and not trades_df.empty:
                    st.subheader("Trade Log")
                    st.dataframe(trades_df)

                details_df = get_detailed_risk_regime_infos(results_all[ticker], ticker)
                if not details_df.empty:
                    st.subheader("Regime Details")
                    st.dataframe(details_df)

                #import plotly.graph_objects as go

                #if strat_a is not None and strat_b is not None:
                #    st.subheader("📈 Strategy Comparison (Interactive)")

                    # Convert to Series if necessary
                #    if isinstance(strat_a, pd.DataFrame) and 'Equity Curve' in strat_a.columns:
                #        strat_a_series = strat_a['Equity Curve']
                #    else:
                #        strat_a_series = strat_a

                #    if isinstance(strat_b, pd.DataFrame) and 'Equity Curve' in strat_b.columns:
                #        strat_b_series = strat_b['Equity Curve']
                #    else:
                #        strat_b_series = strat_b

                #    fig = go.Figure()
                #    fig.add_trace(go.Scatter(
                #        x=strat_a_series.index,
                #        y=strat_a_series.values,
                #        mode='lines',
                #        name='Strategy A',
                #        line=dict(color='blue')
                #    ))
                #    fig.add_trace(go.Scatter(
                #        x=strat_b_series.index,
                #        y=strat_b_series.values,
                #        mode='lines',
                #        name='Strategy B',
                #        line=dict(color='orange')
                #    ))

                #    fig.update_layout(
                #        xaxis_title='Date',
                #        yaxis_title='Equity Value',
                #        title=f'{ticker} – Strategy Equity Comparison',
                #        height=300,
                #        margin=dict(t=40, b=30),
                #        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                #    )

                #    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No results for {ticker}")
