import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from requests.exceptions import RequestException

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
MARKET_DATA_START = "2015-01-01"
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30
MONTE_CARLO_SEEDS = {
    "Conservative Income": 101,
    "Balanced Growth": 202,
    "Growth-Oriented": 303
}


def empty_market_news_frame():
    return pd.DataFrame(columns=["title", "source", "url", "time", "summary"])


@st.cache_data(ttl=21600)
def get_market_news():
    if not ALPHA_VANTAGE_API_KEY:
        return empty_market_news_frame()

    params = {
        "function": "NEWS_SENTIMENT",
        "topics": "financial_markets",
        "sort": "LATEST",
        "limit": 8,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params=params,
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
    except (RequestException, ValueError):
        return empty_market_news_frame()


    articles = []
    backup_articles = []

    trusted_sources = [
        "Reuters",
        "CNBC",
        "MarketWatch",
        "Yahoo Finance",
        "Bloomberg",
        "Financial Times",
        "The Wall Street Journal",
        "Investopedia"
    ]

    if "feed" in data:
        for item in data["feed"]:

            source = item.get("source", "Unknown source")

            article = {
                "title": item.get("title", "No title"),
                "source": source,
                "url": item.get("url", ""),
                "time": item.get("time_published", "")[:8],
                "summary": item.get("summary", "")
            }

            if source in trusted_sources:
                articles.append(article)
            else:
                backup_articles.append(article)

            if len(articles) >= 10:
                break

    articles = articles + backup_articles

    if not articles:
        return empty_market_news_frame()

    return pd.DataFrame(articles[:10])

st.set_page_config(page_title="Multi-Asset Portfolio Strategy", layout="wide")

st.title("Multi-Asset Portfolio Strategy Platform")
st.caption(
    "Strategic multi-asset portfolio construction, historical backtesting, "
    "risk analytics, and live cross-asset market monitoring."
)

# ---------------------------------------------------
# Asset Universe
# ---------------------------------------------------

ASSETS = {
    "VTI": "U.S. Total Market",
    "VXUS": "International Equities",
    "QQQ": "U.S. Growth / Technology",
    "SCHD": "Dividend / Quality Equity",
    "BND": "Aggregate Bonds",
    "IEF": "Intermediate Treasuries",
    "GLD": "Gold",
    "VNQ": "Real Estate",
    "BIL": "Treasury Bills"
}

tickers = list(ASSETS.keys())

expected_returns = pd.Series({
    "VTI": 0.095,
    "VXUS": 0.085,
    "QQQ": 0.115,
    "SCHD": 0.085,
    "BND": 0.045,
    "IEF": 0.040,
    "GLD": 0.050,
    "VNQ": 0.075,
    "BIL": 0.035
})

expected_volatility = pd.Series({
    "VTI": 0.16,
    "VXUS": 0.18,
    "QQQ": 0.22,
    "SCHD": 0.14,
    "BND": 0.06,
    "IEF": 0.07,
    "GLD": 0.15,
    "VNQ": 0.20,
    "BIL": 0.01
})

risk_free_rate = 0.035
num_portfolios = 30000

COLOR_MAP = {
    "Selected Portfolio": "#38BDF8",
    "SPY": "#22C55E",
    "SPY Proxy": "#22C55E",
    "60/40": "#F97316",
    "60/40 Portfolio": "#F97316",
    "Conservative Income": "#38BDF8",
    "Balanced Growth": "#22C55E",
    "Growth-Oriented": "#F97316"
}

# ---------------------------------------------------
# Data Functions
# ---------------------------------------------------


def normalize_close_prices(downloaded_data, requested_tickers):
    if downloaded_data is None or downloaded_data.empty:
        return pd.DataFrame(columns=requested_tickers)

    if isinstance(downloaded_data.columns, pd.MultiIndex):
        if "Close" not in downloaded_data.columns.get_level_values(0):
            return pd.DataFrame(columns=requested_tickers)
        close_prices = downloaded_data["Close"]
    elif "Close" in downloaded_data.columns:
        close_prices = downloaded_data["Close"]
    else:
        close_prices = downloaded_data

    if isinstance(close_prices, pd.Series):
        ticker = requested_tickers[0] if len(requested_tickers) == 1 else close_prices.name
        close_prices = close_prices.to_frame(name=ticker)

    close_prices = close_prices.copy()
    close_prices.columns = [str(col) for col in close_prices.columns]
    close_prices = close_prices.reindex(columns=requested_tickers)
    close_prices = close_prices.apply(pd.to_numeric, errors="coerce")
    close_prices = close_prices.replace([np.inf, -np.inf], np.nan)
    close_prices = close_prices.dropna(how="all").ffill()

    return close_prices.dropna(how="all")


def download_close_prices(requested_tickers, start=None, period=None):
    requested_tickers = list(requested_tickers)

    for _ in range(DOWNLOAD_RETRIES):
        try:
            downloaded_data = yf.download(
                requested_tickers,
                start=start,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=DOWNLOAD_TIMEOUT
            )
        except Exception:
            continue

        close_prices = normalize_close_prices(downloaded_data, requested_tickers)

        if not close_prices.empty:
            missing_tickers = [
                ticker for ticker in requested_tickers
                if ticker not in close_prices.columns or close_prices[ticker].dropna().empty
            ]

            if missing_tickers:
                fallback_prices = download_chart_api_prices(missing_tickers, start=start, period=period)
                close_prices = close_prices.combine_first(fallback_prices)

            return close_prices.reindex(columns=requested_tickers)

    return download_chart_api_prices(requested_tickers, start=start, period=period)


def download_chart_api_prices(requested_tickers, start=None, period=None):
    price_series = {}

    for ticker in requested_tickers:
        series = download_chart_api_price(ticker, start=start, period=period)

        if not series.empty:
            price_series[ticker] = series

    if not price_series:
        return pd.DataFrame(columns=requested_tickers)

    prices = pd.DataFrame(price_series)
    prices = prices.reindex(columns=requested_tickers)
    prices = prices.replace([np.inf, -np.inf], np.nan)

    return prices.dropna(how="all").ffill()


def download_chart_api_price(ticker, start=None, period=None):
    end_timestamp = int(pd.Timestamp.utcnow().timestamp())

    if start is not None:
        start_timestamp = int(pd.Timestamp(start).timestamp())
    elif period == "1y":
        start_timestamp = int((pd.Timestamp.utcnow() - pd.DateOffset(years=1)).timestamp())
    else:
        start_timestamp = int((pd.Timestamp.utcnow() - pd.DateOffset(years=10)).timestamp())

    params = {
        "period1": start_timestamp,
        "period2": end_timestamp,
        "interval": "1d",
        "includeAdjustedClose": "true"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        try:
            response = requests.get(
                f"https://{host}/v8/finance/chart/{ticker}",
                params=params,
                headers=headers,
                timeout=DOWNLOAD_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
        except (RequestException, ValueError):
            continue

        result = data.get("chart", {}).get("result")

        if not result:
            continue

        timestamps = result[0].get("timestamp", [])
        indicators = result[0].get("indicators", {})
        adjclose = indicators.get("adjclose", [{}])[0].get("adjclose")
        close = indicators.get("quote", [{}])[0].get("close")
        values = adjclose if adjclose is not None else close

        if not timestamps or not values:
            continue

        series = pd.Series(
            values,
            index=pd.to_datetime(timestamps, unit="s"),
            name=ticker
        )
        series = pd.to_numeric(series, errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        if not series.empty:
            return series

    return pd.Series(dtype=float, name=ticker)


def clean_returns(prices):
    if prices.empty:
        return pd.DataFrame(columns=prices.columns)

    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    return returns.dropna(axis=1, how="all")


def build_correlation_matrix(returns, tickers):
    if returns.empty:
        return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)

    corr = returns.corr().reindex(index=tickers, columns=tickers)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr = corr.fillna(0.0).clip(lower=-1.0, upper=1.0)
    np.fill_diagonal(corr.values, 1.0)

    return corr


@st.cache_data(ttl=86400)
def load_data(tickers):
    prices = download_close_prices(tickers, start=MARKET_DATA_START)
    returns = clean_returns(prices).reindex(columns=tickers)
    historical_corr = build_correlation_matrix(returns, tickers)

    return prices, returns, historical_corr


@st.cache_data(ttl=3600)
def load_market_snapshot(market_tickers):
    data = download_close_prices(market_tickers.values(), period="1y")

    rows = []

    for name, ticker in market_tickers.items():
        if ticker not in data.columns:
            continue

        series = data[ticker].dropna()

        if len(series) < 2:
            continue

        latest = series.iloc[-1]
        previous = series.iloc[-2]
        start_year = series[series.index.year == series.index[-1].year].iloc[0]

        one_day_change = (latest / previous) - 1
        ytd_change = (latest / start_year) - 1

        rows.append({
            "Market Indicator": name,
            "Ticker": ticker,
            "Latest": latest,
            "1D Change": one_day_change,
            "YTD Change": ytd_change
        })

    return pd.DataFrame(
        rows,
        columns=["Market Indicator", "Ticker", "Latest", "1D Change", "YTD Change"]
    )


@st.cache_data(ttl=86400)
def load_benchmark_returns(benchmark_tickers):
    benchmark_prices = download_close_prices(
        benchmark_tickers,
        start=MARKET_DATA_START
    )
    benchmark_returns = clean_returns(benchmark_prices).reindex(columns=benchmark_tickers)

    return benchmark_returns


def split_with_caps(total_weight, assets, max_single_weight, rng):
    for _ in range(1000):
        raw = rng.dirichlet(np.ones(len(assets)))
        weights = raw * total_weight

        if weights.max() <= max_single_weight:
            return dict(zip(assets, weights))

    equal_weight = total_weight / len(assets)
    return dict(zip(assets, [equal_weight] * len(assets)))


@st.cache_data(ttl=86400)
def run_simulation(profile, expected_returns, strategic_cov_matrix, tickers, num_portfolios, risk_free_rate):
    results = []
    rng = np.random.default_rng(MONTE_CARLO_SEEDS[profile])
    expected_returns = expected_returns.reindex(tickers).astype(float)
    cov_values = strategic_cov_matrix.reindex(index=tickers, columns=tickers).values
    cov_values = np.nan_to_num(cov_values, nan=0.0, posinf=0.0, neginf=0.0)

    equity_assets = ["VTI", "VXUS", "QQQ", "SCHD"]
    bond_cash_assets = ["BND", "IEF", "BIL"]
    alternative_assets = ["GLD", "VNQ"]

    for _ in range(num_portfolios):

        if profile == "Conservative Income":
            equity_weight = rng.uniform(0.25, 0.35)
            bond_cash_weight = rng.uniform(0.55, 0.65)

        elif profile == "Balanced Growth":
            equity_weight = rng.uniform(0.50, 0.65)
            bond_cash_weight = rng.uniform(0.25, 0.40)

        else:
            equity_weight = rng.uniform(0.82, 0.92)
            bond_cash_weight = rng.uniform(0.03, 0.10)

        alternatives_weight = 1 - equity_weight - bond_cash_weight

        if not (0.05 <= alternatives_weight <= 0.18):
            continue

        weights_dict = {}
        weights_dict.update(split_with_caps(equity_weight, equity_assets, 0.35, rng))
        weights_dict.update(split_with_caps(bond_cash_weight, bond_cash_assets, 0.35, rng))
        weights_dict.update(split_with_caps(alternatives_weight, alternative_assets, 0.35, rng))

        weights = np.array([weights_dict[t] for t in tickers])

        if weights.max() > 0.35:
            continue

        if weights_dict["VXUS"] < 0.08:
            continue

        if weights_dict["GLD"] + weights_dict["VNQ"] < 0.05:
            continue

        portfolio_return = np.dot(weights, expected_returns.values)

        portfolio_variance = np.dot(weights.T, np.dot(cov_values, weights))

        if not np.isfinite(portfolio_return) or not np.isfinite(portfolio_variance):
            continue

        if portfolio_variance <= 0:
            continue

        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

        if not np.isfinite(sharpe):
            continue

        results.append({
            "Expected Return": portfolio_return,
            "Expected Volatility": portfolio_volatility,
            "Sharpe Ratio": sharpe,
            "Equity Weight": equity_weight,
            "Bond/Cash Weight": bond_cash_weight,
            "Alternatives Weight": alternatives_weight,
            **weights_dict
        })

    return pd.DataFrame(results)


def clean_monte_carlo_results(mc_df):
    if mc_df.empty:
        return mc_df

    numeric_cols = [
        "Expected Return",
        "Expected Volatility",
        "Sharpe Ratio",
        "Equity Weight",
        "Bond/Cash Weight",
        "Alternatives Weight",
        *tickers
    ]

    clean_df = mc_df.copy()
    clean_df[numeric_cols] = clean_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)

    finite_mask = np.isfinite(clean_df[numeric_cols]).all(axis=1)
    clean_df = clean_df.loc[finite_mask]
    clean_df = clean_df[clean_df["Expected Volatility"] > 0]

    return clean_df


def select_profile_portfolio(mc_df, profile):
    valid_df = clean_monte_carlo_results(mc_df)

    if valid_df.empty:
        return None

    if profile == "Conservative Income":
        low_vol_cutoff = valid_df["Expected Volatility"].quantile(0.35)
        candidates = valid_df[valid_df["Expected Volatility"] <= low_vol_cutoff]

    elif profile == "Balanced Growth":
        lower = valid_df["Expected Volatility"].quantile(0.35)
        upper = valid_df["Expected Volatility"].quantile(0.75)

        candidates = valid_df[
            (valid_df["Expected Volatility"] >= lower) &
            (valid_df["Expected Volatility"] <= upper)
        ]

    else:
        high_vol_cutoff = valid_df["Expected Volatility"].quantile(0.75)
        candidates = valid_df[valid_df["Expected Volatility"] >= high_vol_cutoff]

    if candidates.empty:
        candidates = valid_df

    selected = candidates.sort_values(
        ["Sharpe Ratio", "Expected Return"],
        ascending=[False, False],
        kind="mergesort"
    ).iloc[0]

    return selected


def performance_metrics(r, risk_free_rate):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()

    if r.empty:
        return [np.nan, np.nan, np.nan, np.nan]

    annual_return = (1 + r.mean()) ** 252 - 1
    annual_vol = r.std() * np.sqrt(252)

    if np.isfinite(annual_vol) and annual_vol > 0:
        sharpe = (annual_return - risk_free_rate) / annual_vol
    else:
        sharpe = np.nan

    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return [annual_return, annual_vol, sharpe, max_dd]


def calculate_portfolio_returns(returns, weights):
    weights = pd.to_numeric(weights, errors="coerce")
    returns = returns.reindex(columns=weights.index).replace([np.inf, -np.inf], np.nan)

    usable_assets = returns.columns[
        returns.notna().any() &
        weights.replace([np.inf, -np.inf], np.nan).notna()
    ]

    if len(usable_assets) == 0:
        return pd.Series(dtype=float)

    usable_weights = weights.loc[usable_assets]
    usable_weights = usable_weights[usable_weights > 0]

    if usable_weights.empty or usable_weights.sum() <= 0:
        return pd.Series(dtype=float)

    usable_weights = usable_weights / usable_weights.sum()
    usable_returns = returns.loc[:, usable_weights.index]
    portfolio_returns = usable_returns.mul(usable_weights, axis=1).sum(axis=1, min_count=1)

    return portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_growth_and_drawdown(portfolio_returns):
    if portfolio_returns.empty:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series

    growth = (1 + portfolio_returns).cumprod() * 10000
    drawdown = growth / growth.cummax() - 1

    return growth, drawdown


def get_return_series(returns, ticker):
    if ticker not in returns.columns:
        return pd.Series(dtype=float)

    return returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()


def prepare_chart_dataframe(data):
    chart_df = pd.DataFrame(data)
    chart_df = chart_df.apply(pd.to_numeric, errors="coerce")

    return chart_df.dropna(how="all")


def has_chart_data(chart_df):
    return not chart_df.empty and chart_df.notna().any().any()


def style_map(styler, func, subset):
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)

    return styler.applymap(func, subset=subset)


def is_finite_number(value):
    return pd.notna(value) and np.isfinite(value)


def format_percent(value):
    if not is_finite_number(value):
        return "N/A"

    return f"{value:.2%}"


def format_number(value):
    if not is_finite_number(value):
        return "N/A"

    return f"{value:.2f}"


# ---------------------------------------------------
# Load Base Data
# ---------------------------------------------------

prices, returns, historical_corr = load_data(tickers)

strategic_cov_matrix = pd.DataFrame(
    np.outer(expected_volatility, expected_volatility) * historical_corr.values,
    index=tickers,
    columns=tickers
)

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Portfolio Analysis", "Live Markets", "Investor Profile"]
)

st.sidebar.markdown("---")

if page == "Portfolio Analysis":
    profile = st.sidebar.radio(
        "Select Portfolio Type",
        ["Conservative Income", "Balanced Growth", "Growth-Oriented"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Selected Mandate")

    if profile == "Conservative Income":
        st.sidebar.write("Capital preservation, income, and lower volatility.")
    elif profile == "Balanced Growth":
        st.sidebar.write("Balanced long-term growth with controlled risk.")
    else:
        st.sidebar.write("Long-term capital appreciation with higher equity exposure.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Framework")
    st.sidebar.caption(
        "Strategic assumptions, historical correlations, diversification limits, and risk-profile-specific selection."
    )

elif page == "Live Markets":
    st.sidebar.subheader("Live Markets")
    st.sidebar.write("Current market conditions, cross-asset performance, and relevant news.")

else:
    st.sidebar.subheader("Investor Profile")
    st.sidebar.write("Questionnaire to identify the most suitable portfolio strategy.")

    

# ---------------------------------------------------
# Portfolio Analysis Page
# ---------------------------------------------------

if page == "Portfolio Analysis":

    mc_df = run_simulation(
        profile,
        expected_returns,
        strategic_cov_matrix,
        tickers,
        num_portfolios,
        risk_free_rate
    )
    mc_df = clean_monte_carlo_results(mc_df)

    if mc_df.empty:
        st.error("No valid portfolios matched the current constraints. Try refreshing the app.")
        st.stop()

    selected = select_profile_portfolio(mc_df, profile)
    if selected is None:
        st.error("No valid portfolios matched the current constraints. Try refreshing the app.")
        st.stop()

    weights = pd.to_numeric(selected[tickers], errors="coerce")

    if weights.isna().any() or not np.isfinite(weights).all() or weights.sum() <= 0:
        st.error("The selected portfolio weights could not be calculated. Try refreshing the app.")
        st.stop()

    weights = weights / weights.sum()

    portfolio_returns = calculate_portfolio_returns(returns, weights)
    growth, drawdown = calculate_growth_and_drawdown(portfolio_returns)

    rolling_vol = portfolio_returns.rolling(63).std() * np.sqrt(252)
    rolling_return = portfolio_returns.rolling(252).apply(lambda x: (1 + x).prod() - 1)

    st.header(profile)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        with st.container(border=True):
            st.metric("Expected Annual Return", format_percent(selected["Expected Return"]))

    with m2:
        with st.container(border=True):
            st.metric("Expected Volatility", format_percent(selected["Expected Volatility"]))

    with m3:
        with st.container(border=True):
            st.metric("Sharpe Ratio", format_number(selected["Sharpe Ratio"]))

    with m4:
        with st.container(border=True):
            st.metric("Historical Max Drawdown", format_percent(drawdown.min()))

    tab0, tab1, tab2, tab3, tab_stress, tab_outlook, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Portfolio Construction",
    "Optimization Framework",
    "Historical Backtest",
    "Stress Testing",
    "Forward-Looking Outlook",
    "Risk Analysis",
    "Assumptions"
])

    with tab0:
        st.subheader("Investment Objective")

        if profile == "Conservative Income":
            st.write(
                "The Conservative Income portfolio is designed for investors who prioritize capital preservation, "
                "income, and downside protection over maximum long-term growth. The portfolio maintains a defensive "
                "allocation through Treasury bills, aggregate bonds, and moderate equity exposure."
            )

            st.info(
                "Best suited for: retirees, near-retirees, income-focused investors, or clients with lower risk tolerance."
            )

        elif profile == "Balanced Growth":
            st.write(
                "The Balanced Growth portfolio is designed as a core long-term wealth allocation. It seeks to balance "
                "capital appreciation with risk control by combining equities, bonds, cash, and real assets."
            )

            st.info(
                "Best suited for: long-term investors seeking growth while still managing downside risk."
            )

        else:
            st.write(
                "The Growth-Oriented portfolio is designed for investors with a longer time horizon and greater tolerance "
                "for volatility. It emphasizes equity exposure and long-term compounding while still enforcing diversification limits."
            )

            st.info(
                "Best suited for: younger investors, long-horizon investors, or clients prioritizing capital appreciation."
            )
        st.markdown("---")
        st.subheader("Key Portfolio Takeaway")

        c1, c2, c3 = st.columns(3)

        with c1:
            with st.container(border=True):
                st.metric("Equity Exposure", f"{selected['Equity Weight']:.2%}")

        with c2:
            with st.container(border=True):
                st.metric("Bond / Cash Exposure", f"{selected['Bond/Cash Weight']:.2%}")

        with c3:
            with st.container(border=True):
                st.metric("Alternatives Exposure", f"{selected['Alternatives Weight']:.2%}")

        st.write(
            f"""
            This portfolio targets an expected annual return of **{format_percent(selected['Expected Return'])}**
            with expected volatility of **{format_percent(selected['Expected Volatility'])}**.
            The historical max drawdown for this selected allocation was **{format_percent(drawdown.min())}**.
            """
        )
        st.markdown("---")
        st.subheader("Strategic Positioning")

        positioning = pd.DataFrame({
            "Theme": [
                "Growth",
                "Income / Stability",
                "Inflation Sensitivity",
                "Downside Protection",
                "Diversification"
            ],
            "Portfolio Role": [
                "Equity ETFs provide long-term capital appreciation.",
                "Bonds and Treasury bills help stabilize portfolio volatility.",
                "Gold and real estate add real asset exposure.",
                "Cash, bonds, and diversification reduce reliance on equities alone.",
                "Allocation limits prevent excessive concentration in a single ETF."
            ]
        })

        st.dataframe(positioning, hide_index=True, use_container_width=True)

    with tab1:
        st.subheader("Strategic Allocation")

        allocation_df = pd.DataFrame({
            "Ticker": tickers,
            "Asset Class": [ASSETS[t] for t in tickers],
            "Allocation (%)": (weights.values * 100).round(2)
        })

        c1, c2 = st.columns([1, 1])

        with c1:
            st.dataframe(allocation_df, hide_index=True, use_container_width=True)

        with c2:
            fig = px.pie(
                allocation_df,
                names="Ticker",
                values="Allocation (%)",
                hole=0.4,
                title="Portfolio Allocation"
            )

            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Allocation Summary")

        summary_df = pd.DataFrame({
            "Category": ["Equities", "Bonds / Cash", "Alternatives"],
            "Weight": [
                selected["Equity Weight"],
                selected["Bond/Cash Weight"],
                selected["Alternatives Weight"]
            ]
        })

        summary_df["Weight"] = summary_df["Weight"].map(lambda x: f"{x:.2%}")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        st.markdown("---")
        st.subheader("Portfolio Philosophy")

        if profile == "Conservative Income":
            st.write(
                "This portfolio prioritizes capital preservation, income, and lower volatility. "
                "It emphasizes bonds, Treasuries, and cash-like exposure while maintaining moderate equity exposure "
                "for purchasing power and long-term compounding."
            )

        elif profile == "Balanced Growth":
            st.write(
                "This portfolio balances long-term appreciation with risk control. "
                "It combines meaningful equity exposure with bonds, cash, and real assets to provide diversified return sources "
                "across different market environments."
            )

        else:
            st.write(
                "This portfolio emphasizes long-term capital appreciation through elevated equity exposure. "
                "The selection process still prioritizes risk-adjusted efficiency within a growth-oriented risk band, "
                "rather than simply maximizing raw expected return."
            )

    with tab2:
        st.subheader("Portfolio Construction Constraints")

        if profile == "Conservative Income":
            constraints_df = pd.DataFrame({
                "Constraint": [
                    "Equity allocation range",
                    "Bond / cash allocation range",
                    "Alternatives allocation range",
                    "Maximum single ETF weight",
                    "Minimum international equity exposure",
                    "Portfolio fully invested"
                ],
                "Rule": [
                    "25% - 35%",
                    "55% - 65%",
                    "5% - 18%",
                    "35%",
                    "8%",
                    "Weights sum to 100%"
                ]
            })

        elif profile == "Balanced Growth":
            constraints_df = pd.DataFrame({
                "Constraint": [
                    "Equity allocation range",
                    "Bond / cash allocation range",
                    "Alternatives allocation range",
                    "Maximum single ETF weight",
                    "Minimum international equity exposure",
                    "Portfolio fully invested"
                ],
                "Rule": [
                    "50% - 65%",
                    "25% - 40%",
                    "5% - 18%",
                    "35%",
                    "8%",
                    "Weights sum to 100%"
                ]
            })

        else:
            constraints_df = pd.DataFrame({
                "Constraint": [
                    "Equity allocation range",
                    "Bond / cash allocation range",
                    "Alternatives allocation range",
                    "Maximum single ETF weight",
                    "Minimum international equity exposure",
                    "Portfolio fully invested"
                ],
                "Rule": [
                    "82% - 92%",
                    "3% - 10%",
                    "5% - 18%",
                    "35%",
                    "8%",
                    "Weights sum to 100%"
                ]
            })

        st.dataframe(
            constraints_df,
            use_container_width=True,
            hide_index=True
        )
        st.markdown("---")
        st.subheader("Monte Carlo Portfolio Universe")

        fig = px.scatter(
            mc_df,
            x="Expected Volatility",
            y="Expected Return",
            color="Sharpe Ratio",
            hover_data=tickers,
            title="Simulated Portfolios Within Strategic Constraints"
        )

        fig.add_trace(
            go.Scatter(
                x=[selected["Expected Volatility"]],
                y=[selected["Expected Return"]],
                mode="markers",
                marker=dict(size=16, color="red", symbol="star"),
                name="Selected Portfolio"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.write(
            "The model does not simply select the unconstrained maximum-Sharpe portfolio. "
            "Each portfolio is first constrained by strategic allocation rules, then selected using objectives that match its mandate. "
            "Conservative portfolios prioritize lower volatility, Balanced portfolios target moderate risk-adjusted efficiency, "
            "and Growth portfolios select the strongest Sharpe ratio within a higher-risk growth allocation band."
        )

       


    with tab3:
        st.subheader("Historical Portfolio Backtest")

        st.markdown(
            """
            Backtest compares the selected strategic allocation against common benchmark portfolios
            using historical ETF returns.
            """
        )

        benchmark_tickers = ["SPY", "AGG"]
        benchmark_returns = load_benchmark_returns(benchmark_tickers)

        portfolio_returns_bt = calculate_portfolio_returns(returns, weights)

        spy_returns = get_return_series(benchmark_returns, "SPY")
        sixty_forty_returns = calculate_portfolio_returns(
            benchmark_returns,
            pd.Series({"SPY": 0.60, "AGG": 0.40})
        )

        portfolio_growth = (1 + portfolio_returns_bt).cumprod() * 10000
        spy_growth = (1 + spy_returns).cumprod() * 10000
        sixty_growth = (1 + sixty_forty_returns).cumprod() * 10000

        growth_df = prepare_chart_dataframe({
            "Selected Portfolio": portfolio_growth,
            "SPY": spy_growth,
            "60/40 Portfolio": sixty_growth
        })

        if has_chart_data(growth_df):
            fig_growth = px.line(
                growth_df,
                title="Growth of $10,000",
                color_discrete_map={
            "Selected Portfolio": "#38BDF8",
            "SPY": "#22C55E",
            "60/40 Portfolio": "#F97316"
        }
    )

            st.plotly_chart(fig_growth, use_container_width=True)
        else:
            st.warning("Historical growth data is temporarily unavailable. Try refreshing the app.")

        metrics_df = pd.DataFrame(
            {
                "Selected Portfolio": performance_metrics(portfolio_returns_bt, risk_free_rate),
                "SPY": performance_metrics(spy_returns, risk_free_rate),
                "60/40 Portfolio": performance_metrics(sixty_forty_returns, risk_free_rate)
            },
            index=[
                "Annual Return",
                "Volatility",
                "Sharpe Ratio",
                "Max Drawdown"
            ]
        ).T

        metrics_display = metrics_df.copy()

        metrics_display["Annual Return"] = metrics_display["Annual Return"].map(format_percent)

        metrics_display["Volatility"] = metrics_display["Volatility"].map(format_percent)

        metrics_display["Sharpe Ratio"] = metrics_display["Sharpe Ratio"].map(format_number)

        metrics_display["Max Drawdown"] = metrics_display["Max Drawdown"].map(format_percent)
        st.markdown("---")
        st.subheader("Benchmark Comparison")

        def color_positive_negative_pct(val):
            if not isinstance(val, str) or not val.endswith("%"):
                return ""

            if val.startswith("-"):
                return "color: #EF4444; font-weight: 700;"
            return "color: #22C55E; font-weight: 700;"


        drawdown_values = pd.to_numeric(
            metrics_display["Max Drawdown"].str.replace("%", ""),
            errors="coerce"
        )

        drawdown_rank = drawdown_values.rank(method="first", ascending=True)
        drawdown_color_map = {}
        drawdown_palette = ["#991B1B", "#EF4444", "#FCA5A5"]

        for rank, color in enumerate(drawdown_palette, start=1):
            ranked_index = drawdown_rank.index[drawdown_rank == rank]
            if len(ranked_index) > 0:
                drawdown_color_map[ranked_index[0]] = color

        def color_drawdown_relative(val):
            if val == "N/A":
                return ""

            matches = metrics_display[metrics_display["Max Drawdown"] == val]
            if matches.empty:
                return ""

            row_label = matches.index[0]
            color = drawdown_color_map.get(row_label, "#EF4444")
            return f"color: {color}; font-weight: 800;"


        styled_metrics = style_map(
            style_map(
                metrics_display.style,
                color_positive_negative_pct,
                subset=["Annual Return", "Volatility"]
            ),
            color_drawdown_relative,
            subset=["Max Drawdown"]
        )

        st.dataframe(
            styled_metrics,
            use_container_width=True
        )

        st.info(
            "Benchmark comparison helps evaluate whether the selected allocation improves risk-adjusted returns "
            "versus traditional market exposures."
        )
        st.markdown("---")
        st.subheader("Historical Drawdown Comparison")

        drawdown_comparison = growth_df / growth_df.cummax() - 1

        if has_chart_data(drawdown_comparison):
            fig_drawdown = px.line(
                drawdown_comparison,
                title="Drawdown Comparison",
                color_discrete_map=COLOR_MAP
            )

            st.plotly_chart(fig_drawdown, use_container_width=True)
        else:
            st.warning("Drawdown comparison data is temporarily unavailable. Try refreshing the app.")
        st.markdown("---")
        st.subheader("Rolling Volatility Comparison")

        rolling_vol_comparison = growth_df.pct_change().rolling(63).std() * np.sqrt(252)

        if has_chart_data(rolling_vol_comparison):
            fig_rolling_vol = px.line(
                rolling_vol_comparison,
                title="Rolling 3-Month Annualized Volatility",
                color_discrete_map=COLOR_MAP
            )
        
            st.plotly_chart(fig_rolling_vol, use_container_width=True)
        else:
            st.warning("Rolling volatility comparison data is temporarily unavailable. Try refreshing the app.")
        st.markdown("---")
        st.subheader("Rolling 12-Month Return")

        rolling_return_df = pd.DataFrame({
            "Date": rolling_return.index,
            "Rolling 12-Month Return": rolling_return.values
        }).dropna()

        if has_chart_data(rolling_return_df[["Rolling 12-Month Return"]]):
            fig = px.line(
                rolling_return_df,
                x="Date",
                y="Rolling 12-Month Return",
                title="Rolling 12-Month Portfolio Return",
                color_discrete_map=COLOR_MAP
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Rolling 12-month return data is temporarily unavailable. Try refreshing the app.")
        st.markdown("---")
        st.subheader("Annual Return Comparison")

        if has_chart_data(growth_df) and isinstance(growth_df.index, pd.DatetimeIndex):
            annual_returns_comparison = growth_df.pct_change().resample("YE").apply(
                lambda x: (1 + x).prod() - 1
            )

            annual_returns_comparison.index = annual_returns_comparison.index.year

            annual_returns_display = annual_returns_comparison.copy()

            for col in annual_returns_display.columns:
                annual_returns_display[col] = annual_returns_display[col].map(format_percent)

            def color_annual_returns(val):
                if isinstance(val, str) and val.startswith("-"):
                    return "color: #EF4444; font-weight: 700;"
                if isinstance(val, str) and val.endswith("%"):
                    return "color: #22C55E; font-weight: 700;"
                return ""

            styled_annual_returns = style_map(
                annual_returns_display.style,
                color_annual_returns,
                subset=["Selected Portfolio", "SPY", "60/40 Portfolio"]
            )

            st.dataframe(
                styled_annual_returns,
                use_container_width=True
            )
        else:
            st.warning("Annual return comparison data is temporarily unavailable. Try refreshing the app.")
        st.markdown("---")
        st.subheader("Risk / Return Positioning")

        risk_return_df = metrics_df.reset_index().rename(columns={"index": "Portfolio"})

        fig_scatter = px.scatter(
            risk_return_df,
            x="Volatility",
            y="Annual Return",
            text="Portfolio",
            size=[20, 20, 20],
            title="Risk vs Return",
            color_discrete_map=COLOR_MAP
        )

        fig_scatter.update_traces(textposition="top center")

        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_stress:
        st.subheader("Scenario Stress Testing")

        st.write(
            "This section evaluates how the selected allocation performed across major market environments "
            "relative to SPY and a traditional 60/40 benchmark."
        )

        benchmark_tickers = ["SPY", "AGG"]
        benchmark_returns_stress = load_benchmark_returns(benchmark_tickers)

        spy_returns_stress = get_return_series(benchmark_returns_stress, "SPY")
        sixty_forty_returns_stress = calculate_portfolio_returns(
            benchmark_returns_stress,
            pd.Series({"SPY": 0.60, "AGG": 0.40})
        )

        scenario_periods = {
            "COVID Shock": ("2020-02-19", "2020-04-30"),
            "2022 Inflation / Rate Shock": ("2022-01-03", "2022-10-14"),
            "2023 Risk-On Recovery": ("2023-01-03", "2023-12-29")
        }

        if not portfolio_returns.empty:
            latest_portfolio_date = portfolio_returns.index.max()
            scenario_periods["Last 12 Months"] = (
                str(latest_portfolio_date - pd.DateOffset(months=12))[:10],
                str(latest_portfolio_date)[:10]
            )

        def scenario_stats(return_series, start, end):
            scenario_returns = return_series.loc[start:end]

            if len(scenario_returns) == 0:
                return np.nan, np.nan

            scenario_growth = (1 + scenario_returns).cumprod()
            scenario_return = scenario_growth.iloc[-1] - 1
            scenario_drawdown = scenario_growth / scenario_growth.cummax() - 1
            scenario_max_drawdown = scenario_drawdown.min()

            return scenario_return, scenario_max_drawdown

        scenario_results = []

        for scenario, dates in scenario_periods.items():
            start, end = dates

            portfolio_ret, portfolio_dd = scenario_stats(portfolio_returns, start, end)
            spy_ret, spy_dd = scenario_stats(spy_returns_stress, start, end)
            sixty_ret, sixty_dd = scenario_stats(sixty_forty_returns_stress, start, end)

            scenario_results.append({
                "Scenario": scenario,
                "Start Date": start,
                "End Date": end,
                "Selected Portfolio Return": portfolio_ret,
                "SPY Return": spy_ret,
                "60/40 Return": sixty_ret,
                "Selected Portfolio Max Drawdown": portfolio_dd,
                "SPY Max Drawdown": spy_dd,
                "60/40 Max Drawdown": sixty_dd
            })

        scenario_df = pd.DataFrame(scenario_results)

        display_scenario_df = scenario_df.copy()

        percent_cols = [
            "Selected Portfolio Return",
            "SPY Return",
            "60/40 Return",
            "Selected Portfolio Max Drawdown",
            "SPY Max Drawdown",
            "60/40 Max Drawdown"
        ]

        for col in percent_cols:
            display_scenario_df[col] = display_scenario_df[col].map(format_percent)

        def color_returns(val):
            if not isinstance(val, str) or not val.endswith("%"):
                return ""

            try:
                num = float(val.replace("%", ""))
            except ValueError:
                return ""

            if num >= 0:
                return "color: #22C55E; font-weight: 700;"
            else:
                return "color: #EF4444; font-weight: 700;"


        def color_drawdowns(val):
            if not isinstance(val, str) or not val.endswith("%"):
                return ""

            try:
                num = float(val.replace("%", ""))
            except ValueError:
                return ""

            if num <= -15:
                return "color: #DC2626; font-weight: 800;"
            elif num <= -8:
                return "color: #EF4444; font-weight: 700;"
            else:
                return "color: #F87171; font-weight: 700;"


        styled_scenario_df = style_map(
            style_map(
                display_scenario_df.style,
                color_returns,
                subset=[
                    "Selected Portfolio Return",
                    "SPY Return",
                    "60/40 Return"
                ]
            ),
            color_drawdowns,
            subset=[
                "Selected Portfolio Max Drawdown",
                "SPY Max Drawdown",
                "60/40 Max Drawdown"
            ]
        )

        st.dataframe(
            styled_scenario_df,
            hide_index=True,
            use_container_width=True
        )
        st.markdown("---")
        st.subheader("Scenario Return Comparison")

        scenario_return_chart = scenario_df[
            ["Scenario", "Selected Portfolio Return", "SPY Return", "60/40 Return"]
        ].melt(
            id_vars="Scenario",
            var_name="Portfolio",
            value_name="Return"
        )

        scenario_return_chart["Portfolio"] = scenario_return_chart["Portfolio"].str.replace(" Return", "")

        fig_scenarios = px.bar(
            scenario_return_chart,
            x="Scenario",
            y="Return",
            color="Portfolio",
            barmode="group",
            title="Return Across Market Regimes",
            text="Return",
            color_discrete_map=COLOR_MAP
        )

        fig_scenarios.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
        )

        fig_scenarios.update_layout(
            yaxis_tickformat=".0%",
            yaxis=dict(range=[-0.35, 0.35])
        )

        st.plotly_chart(fig_scenarios, use_container_width=True)
        st.markdown("---")
        st.subheader("Scenario Drawdown Comparison")

        scenario_drawdown_chart = scenario_df[
            ["Scenario", "Selected Portfolio Max Drawdown", "SPY Max Drawdown", "60/40 Max Drawdown"]
        ].melt(
            id_vars="Scenario",
            var_name="Portfolio",
            value_name="Max Drawdown"
        )

        scenario_drawdown_chart["Portfolio"] = (
            scenario_drawdown_chart["Portfolio"]
            .str.replace(" Max Drawdown", "")
        )

        fig_drawdowns = px.bar(
            scenario_drawdown_chart,
            x="Scenario",
            y="Max Drawdown",
            color="Portfolio",
            barmode="group",
            title="Max Drawdown Across Market Regimes",
            text="Max Drawdown",
            color_discrete_map=COLOR_MAP
        )

        fig_drawdowns.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
        )

        fig_drawdowns.update_layout(
            yaxis_tickformat=".0%",
            yaxis=dict(range=[-0.35, 0.05])
        )

        st.plotly_chart(fig_drawdowns, use_container_width=True)
    
    with tab_outlook:
        st.subheader("Forward-Looking Strategic Outlook")

        if profile == "Conservative Income":
            st.write(
                "The Conservative Income portfolio is positioned for investors who prioritize stability, income, "
                "and capital preservation. Going forward, the portfolio is designed to remain resilient across uncertain "
                "market environments by emphasizing Treasury bills, high-quality bonds, and moderate equity exposure."
            )

            st.markdown("---")

            st.subheader("Why This Portfolio Remains Relevant")
            st.write(
                """
                - Higher cash and bond exposure can provide stability if equity volatility rises.
                - Treasury bill exposure can remain attractive in higher-rate environments.
                - Moderate equity exposure helps preserve long-term purchasing power.
                - Gold and real estate provide real-asset diversification against inflation and geopolitical uncertainty.
                """
            )

            st.subheader("Key Risks")
            st.write(
                """
                - Falling rates may reduce future cash income.
                - Persistent inflation could pressure real returns.
                - Lower equity exposure may cause the portfolio to lag during strong bull markets.
                """
            )

        elif profile == "Balanced Growth":
            st.write(
                "The Balanced Growth portfolio is positioned as a core long-term allocation. It is designed to participate "
                "in equity market growth while maintaining enough fixed income, cash, and real asset exposure to help manage "
                "volatility and drawdowns."
            )

            st.markdown("---")

            st.subheader("Why This Portfolio Remains Relevant")
            st.write(
                """
                - Diversified equity exposure provides participation in long-term global growth.
                - Bonds and Treasury bills help reduce reliance on equities alone.
                - Gold and real estate provide exposure to real assets and inflation-sensitive return drivers.
                - The portfolio is designed to be adaptable across growth, inflation, and defensive market regimes.
                """
            )

            st.subheader("Key Risks")
            st.write(
                """
                - Equity market weakness can still create meaningful drawdowns.
                - Rising rates may pressure bond and real estate exposures.
                - International diversification may lag if U.S. markets continue to dominate.
                """
            )

        else:
            st.write(
                "The Growth-Oriented portfolio is positioned for investors with a long time horizon and a higher tolerance "
                "for volatility. Going forward, the portfolio is designed to capture long-term equity risk premia, innovation-led "
                "growth, and compounding potential while still maintaining basic diversification controls."
            )

            st.markdown("---")

            st.subheader("Why This Portfolio Remains Relevant")
            st.write(
                """
                - Higher equity exposure increases participation in long-term market growth.
                - QQQ exposure provides sensitivity to technology, innovation, and productivity themes.
                - International equities provide geographic diversification outside the U.S.
                - Gold and real estate help prevent the portfolio from being purely equity-dependent.
                """
            )

            st.subheader("Key Risks")
            st.write(
                """
                - Higher equity exposure increases drawdown risk.
                - Growth equities can be sensitive to interest rates and valuation compression.
                - This portfolio may underperform defensive allocations during recessions or risk-off markets.
                """
            )

        st.markdown("---")

        st.subheader("Strategic Interpretation")

        outlook_df = pd.DataFrame({
            "Portfolio Lens": [
                "Primary Objective",
                "Forward Return Driver",
                "Defensive Role",
                "Best Market Environment",
                "Main Risk to Monitor"
            ],
            "Interpretation": [
                "Capital preservation" if profile == "Conservative Income" else
                "Balanced long-term growth" if profile == "Balanced Growth" else
                "Long-term capital appreciation",

                "Income, stability, and moderate equity participation" if profile == "Conservative Income" else
                "Global equity growth with diversified risk control" if profile == "Balanced Growth" else
                "Equity risk premium, technology, and innovation-led growth",

                "Bonds, Treasury bills, and lower equity exposure" if profile == "Conservative Income" else
                "Bonds, cash, alternatives, and diversification" if profile == "Balanced Growth" else
                "Diversification limits, real assets, and long investment horizon",

                "Uncertain or defensive markets with attractive cash yields" if profile == "Conservative Income" else
                "Moderate growth environments with balanced risk appetite" if profile == "Balanced Growth" else
                "Risk-on markets with strong earnings and innovation momentum",

                "Inflation and reinvestment risk" if profile == "Conservative Income" else
                "Equity drawdowns and rate-sensitive assets" if profile == "Balanced Growth" else
                "Valuation compression and recession risk"
            ]
        })
        st.dataframe(
            outlook_df,
            hide_index=True,
            use_container_width=True
        )
        market_fit_df = pd.DataFrame({

            "Market Environment": [
                "Risk-On Equity Rally",
                "Rising Rate Environment",
                "Inflation Shock",
                "Recession / Risk-Off",
                "Falling Rate Environment"
            ],
        
            "Expected Portfolio Behavior": [

                "Higher equity exposure should support stronger upside participation."
                if profile == "Growth-Oriented"
                else
                "Diversified equity exposure should allow moderate participation while limiting concentration risk."
                if profile == "Balanced Growth"
                else
                "The portfolio may lag aggressive equity benchmarks but should maintain stability.",

                "Higher yields may pressure bonds and growth equities, though diversification can reduce concentration risk."
                if profile != "Conservative Income"
                else
                "Treasury bill exposure and shorter-duration positioning may help stabilize returns.",

                "Gold and real assets may help offset inflation-related market stress.",

                "Lower equity exposure and diversification may help reduce downside volatility."
                if profile == "Conservative Income"
                else
                "Diversification across equities, bonds, and alternatives may improve resilience."
                if profile == "Balanced Growth"
                else
                "Higher equity exposure may create larger drawdowns during broad risk-off periods.",

                "Bond exposure and duration-sensitive assets may benefit from easing financial conditions."
                if profile != "Growth-Oriented"
                else
                "Growth equities may benefit from lower discount rates and improving liquidity conditions."
            ]
        })
        st.subheader("Market Environment Fit")
        st.dataframe(
            market_fit_df,
            hide_index=True,
            use_container_width=True
        )
        st.markdown("---")
        st.subheader("Chief Investment Office Commentary")

        if profile == "Conservative Income":
            cio_positioning = (
                "The Conservative Income allocation remains focused on capital preservation, income generation, "
                "and lower portfolio volatility. Higher exposure to Treasury bills and high-quality fixed income can "
                "help reduce sensitivity to equity market drawdowns while still allowing measured participation in long-term growth."
            )
            cio_risk = (
                "The main risks to monitor are reinvestment risk if short-term rates decline, inflation pressure on real returns, "
                "and underperformance during strong equity-led rallies."
            )

        elif profile == "Balanced Growth":
            cio_positioning = (
                "The Balanced Growth allocation remains positioned as a core long-term portfolio, balancing equity participation "
                "with fixed income, cash, and real asset diversification. This structure can help investors participate in growth "
                "while reducing reliance on a single market environment."
            )
            cio_risk = (
                "The main risks to monitor are equity drawdowns, rate-sensitive fixed income exposure, and periods where "
                "international or real asset diversification lags U.S. equities."
            )

        else:
            cio_positioning = (
                "The Growth-Oriented allocation remains focused on long-term capital appreciation through higher equity exposure, "
                "including growth and technology-oriented assets. The portfolio is designed for investors who can tolerate higher "
                "near-term volatility in exchange for greater long-term compounding potential."
            )
            cio_risk = (
                "The main risks to monitor are valuation compression, recession risk, and sensitivity to changes in interest rates "
                "that can pressure growth-oriented equities."
            )

        st.info(
            f"""
            **Current View:**  
            This portfolio remains strategically appropriate for its mandate because it aligns the investor objective with a diversified allocation framework.

            **Portfolio Positioning:**  
            {cio_positioning}

            **Risks to Monitor:**  
            {cio_risk}
            """
        )
        st.markdown("---")
        st.subheader("Projected Growth of $10,000")

        projection_years = 10
        initial_investment = 10000

        projected_index = pd.RangeIndex(start=0, stop=projection_years + 1, step=1)

        portfolio_expected_return = selected["Expected Return"]
        spy_expected_return = expected_returns["VTI"]
        sixty_forty_expected_return = (0.60 * expected_returns["VTI"]) + (0.40 * expected_returns["BND"])

        projection_df = pd.DataFrame({
            "Year": projected_index,
            "Selected Portfolio": [
                initial_investment * ((1 + portfolio_expected_return) ** year)
                for year in projected_index
            ],
            "SPY": [
                initial_investment * ((1 + spy_expected_return) ** year)
                for year in projected_index
            ],
            "60/40 Portfolio": [
                initial_investment * ((1 + sixty_forty_expected_return) ** year)
                for year in projected_index
            ]
        })

        fig_projection = px.line(
            projection_df,
            x="Year",
            y=["Selected Portfolio", "SPY", "60/40 Portfolio"],
            title="Projected Growth of $10,000 Over 10 Years",
            color_discrete_map=COLOR_MAP
        )

        fig_projection.update_layout(
            yaxis_title="Projected Portfolio Value",
            xaxis_title="Year"
        )

        st.plotly_chart(fig_projection, use_container_width=True)

        st.caption(
            "Projection uses long-term strategic expected return assumptions and is not a forecast or guarantee of future performance."
        )
        st.markdown("---")
        st.dataframe(
            outlook_df,
            hide_index=True,
            use_container_width=True
        )
    
    with tab4:
        st.subheader("Historical Drawdown")

        drawdown_df = pd.DataFrame({
            "Date": drawdown.index,
            "Drawdown": drawdown.values
        })

        fig = px.line(
            drawdown_df,
            x="Date",
            y="Drawdown",
            title="Portfolio Drawdown",
            color_discrete_map=COLOR_MAP
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Rolling 3-Month Annualized Volatility")

        rolling_vol_df = pd.DataFrame({
            "Date": rolling_vol.index,
            "Rolling Volatility": rolling_vol.values
        }).dropna()

        fig = px.line(
            rolling_vol_df,
            x="Date",
            y="Rolling Volatility",
            title="Rolling 3-Month Annualized Volatility"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Asset Correlation Matrix")

        fig = px.imshow(
            historical_corr.round(2),
            text_auto=True,
            title="Historical Asset Correlation Matrix",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )

        fig.update_layout(
            height=750,
            width=900,
            margin=dict(l=80, r=80, t=80, b=80)
        )

        fig.update_xaxes(tickangle=0)
        fig.update_yaxes(tickangle=0)

        st.plotly_chart(fig, use_container_width=True)


    with tab5:
        st.subheader("Strategic Expected Return and Volatility Assumptions")

        assumptions_df = pd.DataFrame({
            "Ticker": tickers,
            "Asset Class": [ASSETS[t] for t in tickers],
            "Expected Return": [expected_returns[t] for t in tickers],
            "Expected Volatility": [expected_volatility[t] for t in tickers]
        })

        assumptions_df["Expected Return"] = assumptions_df["Expected Return"].map(lambda x: f"{x:.2%}")
        assumptions_df["Expected Volatility"] = assumptions_df["Expected Volatility"].map(lambda x: f"{x:.2%}")

        st.dataframe(assumptions_df, hide_index=True, use_container_width=True)
        st.markdown("---")
        st.write(
            "Expected returns and expected volatility are long-term strategic capital market assumptions, not short-term forecasts. "
            "Historical data is used for correlations, drawdowns, and realized return analysis. "
            "This approach avoids letting one historical market regime dominate the portfolio construction process."
        )

# ---------------------------------------------------
# Live Markets Page
# ---------------------------------------------------

elif page == "Live Markets":
    st.header("Live Market Dashboard")
    st.caption("Current cross-asset market conditions and portfolio-relevant market updates.")

    market_tickers = {
        "S&P 500": "SPY",
        "Nasdaq 100": "QQQ",
        "Dow Jones": "DIA",
        "10Y Treasury Yield": "^TNX",
        "Gold": "GLD",
        "Oil": "USO",
        "U.S. Dollar": "UUP",
        "VIX": "^VIX",
        "Bitcoin": "BTC-USD"
    }

    market_df = load_market_snapshot(market_tickers)

    display_market_df = market_df.copy()
    display_market_df["Latest"] = display_market_df["Latest"].map(
        lambda x: f"{x:,.2f}" if is_finite_number(x) else "N/A"
    )
    display_market_df["1D Change"] = display_market_df["1D Change"].map(format_percent)
    display_market_df["YTD Change"] = display_market_df["YTD Change"].map(format_percent)

    st.subheader("Market Snapshot")

    def color_positive_negative(val):
        if isinstance(val, str) and "-" in val:
            return "color: #EF4444; font-weight: 700;"
        return "color: #22C55E; font-weight: 700;"

    styled_market_df = style_map(
        display_market_df.style,
        color_positive_negative,
        subset=["1D Change", "YTD Change"]
    )

    st.dataframe(
        styled_market_df,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Macro Regime Engine")

    def get_market_value(market_df, indicator, column, default=np.nan):
        match = market_df.loc[
            market_df["Market Indicator"] == indicator,
            column
        ]

        if match.empty:
            return default

        return match.iloc[0]

    sp500_ytd = get_market_value(market_df, "S&P 500", "YTD Change")
    nasdaq_ytd = get_market_value(market_df, "Nasdaq 100", "YTD Change")
    vix_level = get_market_value(market_df, "VIX", "Latest")
    ten_year_ytd = get_market_value(market_df, "10Y Treasury Yield", "YTD Change")
    gold_ytd = get_market_value(market_df, "Gold", "YTD Change")
    oil_ytd = get_market_value(market_df, "Oil", "YTD Change")

    if pd.isna(sp500_ytd) or pd.isna(nasdaq_ytd) or pd.isna(vix_level):
        st.warning("Macro regime engine could not load enough market data. Try refreshing the app.")
    else:
        if sp500_ytd > 0 and nasdaq_ytd > 0 and vix_level < 20:
            risk_regime = "Risk-On"
            risk_comment = "Equities are positive and volatility is contained, suggesting stronger risk appetite."
        elif sp500_ytd < 0 and vix_level >= 20:
            risk_regime = "Risk-Off"
            risk_comment = "Equities are weak and volatility is elevated, suggesting more defensive market conditions."
        else:
            risk_regime = "Mixed / Neutral"
            risk_comment = "Market signals are mixed, suggesting a balanced allocation approach may be appropriate."

        if ten_year_ytd > 0 and oil_ytd > 0:
            macro_regime = "Inflation / Rate Pressure"
            macro_comment = "Rising yields and stronger oil prices suggest inflation or rate pressure may be influencing markets."
        elif ten_year_ytd < 0 and sp500_ytd > 0:
            macro_regime = "Disinflationary Growth"
            macro_comment = "Falling yields alongside positive equity performance suggest a supportive backdrop for risk assets."
        elif gold_ytd > 0 and vix_level >= 20:
            macro_regime = "Defensive / Hedge Demand"
            macro_comment = "Gold strength and elevated volatility suggest investors may be seeking defensive hedges."
        else:
            macro_regime = "Balanced Macro Conditions"
            macro_comment = "No single macro force appears dominant, supporting diversified portfolio positioning."

        r1, r2 = st.columns(2)

        with r1:
            with st.container(border=True):
                st.metric("Risk Regime", risk_regime)
                st.caption(risk_comment)

        with r2:
            with st.container(border=True):
                st.metric("Macro Regime", macro_regime)
                st.caption(macro_comment)

        regime_df = pd.DataFrame({
            "Signal": [
                "Equity Trend",
                "Volatility",
                "Rates",
                "Oil / Inflation",
                "Gold / Defensive Demand"
            ],
            "Current Read": [
                "Positive" if sp500_ytd > 0 else "Negative",
                "Contained" if vix_level < 20 else "Elevated",
                "Rising" if ten_year_ytd > 0 else "Falling",
                "Rising" if oil_ytd > 0 else "Falling",
                "Positive" if gold_ytd > 0 else "Negative"
            ],
            "Interpretation": [
                risk_comment,
                "Low volatility supports risk-taking." if vix_level < 20 else "High volatility supports defensive positioning.",
                "Higher yields can pressure bonds and long-duration equities." if ten_year_ytd > 0 else "Lower yields can support bonds and growth assets.",
                "Higher oil can increase inflation pressure." if oil_ytd > 0 else "Lower oil can reduce inflation pressure.",
                "Gold strength may reflect inflation concern or defensive demand." if gold_ytd > 0 else "Weak gold suggests less demand for defensive hedges."
            ]
        })

        def color_regime_signal(val):
            if val in ["Positive", "Contained"]:
                return "color: #22C55E; font-weight: 700;"
            elif val in ["Negative", "Elevated"]:
                return "color: #EF4444; font-weight: 700;"
            elif val in ["Rising", "Falling"]:
                return "color: #F59E0B; font-weight: 700;"
            return ""

        styled_regime_df = style_map(
            regime_df.style,
            color_regime_signal,
            subset=["Current Read"]
        )

        st.dataframe(
            styled_regime_df,
            hide_index=True,
            use_container_width=True
        )

        st.info(
            f"Current regime read: **{risk_regime} / {macro_regime}**. "
            f"{risk_comment} {macro_comment}"
        )

    st.subheader("Year-to-Date Market Performance")

    fig = px.bar(
        market_df,
        x="Market Indicator",
        y="YTD Change",
        title="YTD Performance by Market Indicator"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Market Read-Through")

    equity_return = get_market_value(market_df, "S&P 500", "YTD Change")
    vix_level = get_market_value(market_df, "VIX", "Latest")
    ten_year_change = get_market_value(market_df, "10Y Treasury Yield", "YTD Change")

    if pd.isna(equity_return) or pd.isna(vix_level) or pd.isna(ten_year_change):
        st.warning("Market read-through could not load enough market data. Try refreshing the app.")
    elif equity_return > 0 and vix_level < 20:
        st.info(
            "Current market conditions appear broadly risk-on, with positive equity performance "
            "and relatively contained volatility. Growth-oriented portfolios may benefit most in this environment."
        )
    elif equity_return < 0 and vix_level > 20:
        st.warning(
            "Current market conditions appear more defensive, with weaker equity performance "
            "and elevated volatility. Conservative and balanced portfolios may provide more downside resilience."
        )
    else:
        st.info(
            "Current market conditions appear mixed, suggesting investors may benefit from diversified exposure "
            "across equities, fixed income, and real assets."
        )

    st.subheader("Cross-Asset Interpretation")

    if pd.isna(equity_return) or pd.isna(vix_level) or pd.isna(ten_year_change):
        interpretation = pd.DataFrame({
            "Signal": ["Portfolio Implication"],
            "Current Read": [
                "Live market data is temporarily incomplete; diversified portfolio positioning remains the strategic baseline."
            ]
        })
    else:
        interpretation = pd.DataFrame({
            "Signal": [
                "Equities",
                "Rates",
                "Volatility",
                "Real Assets",
                "Portfolio Implication"
            ],
            "Current Read": [
                "Positive YTD equity performance supports risk appetite." if equity_return > 0 else "Negative YTD equity performance suggests weaker risk appetite.",
                "Rising Treasury yields may pressure bond prices and rate-sensitive equities." if ten_year_change > 0 else "Falling Treasury yields may support bond prices and long-duration assets.",
                "VIX below 20 suggests calmer markets." if vix_level < 20 else "VIX above 20 suggests elevated uncertainty.",
                "Gold, oil, and real assets help monitor inflation and geopolitical sensitivity.",
                "Diversified portfolios can help balance growth, income, inflation exposure, and downside control."
            ]
        })

    st.dataframe(
        interpretation,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Portfolio-Relevant Market News")

    news_df = get_market_news()

    if news_df.empty:
        st.warning("No news articles were returned. Alpha Vantage may be temporarily unavailable or rate limited.")
    else:
        for _, article in news_df.iterrows():
            summary = article["summary"]

            if len(summary) > 300:
                summary = summary[:300] + "..."

            with st.container(border=True):
                st.markdown(f"### {article['title']}")
                raw_date = str(article.get("time", ""))

                if len(raw_date) == 8:
                    formatted_date = f"{raw_date[4:6]}/{raw_date[6:8]}/{raw_date[0:4]}"
                else:
                    formatted_date = raw_date

                st.caption(f"{article['source']} | {formatted_date}")
                st.write(summary)
                st.markdown(f"[Read full article]({article['url']})")
elif page == "Investor Profile":

    st.header("Investor Profile Questionnaire")
    st.caption(
        "Answer a few questions to estimate which portfolio strategy best fits the investor's objectives and risk tolerance."
    )

    time_horizon = st.selectbox(
        "Investment time horizon",
        [
            "Less than 3 years",
            "3 to 7 years",
            "7 to 15 years",
            "15+ years"
        ]
    )

    risk_tolerance = st.selectbox(
        "How would the investor react to a 20% portfolio decline?",
        [
            "Sell most risky assets",
            "Reduce risk somewhat",
            "Stay invested",
            "Add more capital"
        ]
    )

    income_need = st.selectbox(
        "Current income need",
        [
            "High income need",
            "Moderate income need",
            "Low income need",
            "No income need"
        ]
    )

    investment_goal = st.selectbox(
        "Primary investment objective",
        [
            "Preserve capital",
            "Generate income",
            "Balanced growth",
            "Maximize long-term growth"
        ]
    )

    score = 0

    score += {
        "Less than 3 years": 0,
        "3 to 7 years": 1,
        "7 to 15 years": 2,
        "15+ years": 3
    }[time_horizon]

    score += {
        "Sell most risky assets": 0,
        "Reduce risk somewhat": 1,
        "Stay invested": 2,
        "Add more capital": 3
    }[risk_tolerance]

    score += {
        "High income need": 0,
        "Moderate income need": 1,
        "Low income need": 2,
        "No income need": 3
    }[income_need]

    score += {
        "Preserve capital": 0,
        "Generate income": 1,
        "Balanced growth": 2,
        "Maximize long-term growth": 3
    }[investment_goal]

    if score <= 4:
        recommended_profile = "Conservative Income"
        explanation = "The investor appears to prioritize stability, income, and downside protection."
    elif score <= 8:
        recommended_profile = "Balanced Growth"
        explanation = "The investor appears to have a moderate risk profile and may benefit from balanced exposure."
    else:
        recommended_profile = "Growth-Oriented"
        explanation = "The investor appears suited for higher equity exposure and long-term capital appreciation."

    st.subheader("Recommended Portfolio Strategy")

    st.metric("Suggested Portfolio", recommended_profile)
    st.write(explanation)

    st.info(
        "This questionnaire is a simplified suitability screen for demonstration purposes. "
        "A real advisory process would also consider liquidity needs, taxes, estate planning, employment risk, "
        "concentrated positions, and legal suitability requirements."
    )
