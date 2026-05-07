# Wealth Management Portfolio Analytics Platform

## Overview

This project is an interactive wealth management and portfolio analytics platform built with Python and Streamlit. The application combines portfolio construction, Monte Carlo optimization, historical backtesting, stress testing, macroeconomic analysis, and forward-looking investment commentary into a single dashboard.

The goal of the platform is to simulate how a modern wealth management or investment advisory firm may design, analyze, and communicate portfolio strategies across different investor risk profiles.

The application includes three model portfolio strategies:

- Conservative Income
- Balanced Growth
- Growth-Oriented

Each portfolio is constructed using diversified multi-asset allocations across:
- U.S. equities
- International equities
- Bonds
- Treasury bills
- Gold
- Real estate

---

# Key Features

## Portfolio Construction Framework
- Strategic multi-asset allocation design
- Portfolio-specific allocation constraints
- Monte Carlo portfolio optimization
- Efficient frontier visualization
- Sharpe ratio optimization

## Historical Backtesting
- Growth of $10,000 analysis
- Benchmark comparison vs SPY and 60/40 portfolio
- Rolling performance analysis
- Annual return comparison
- Drawdown analysis
- Volatility comparison

## Stress Testing
Portfolio behavior across major market environments:
- COVID Shock
- Inflation / Rate Shock
- Risk-On Recovery
- Recent market environment

Includes:
- scenario return comparison
- benchmark comparison
- drawdown analysis

## Live Market Dashboard
- Cross-asset market monitoring
- Equity, rates, commodities, and volatility tracking
- Macro regime engine
- Portfolio-relevant market news
- Dynamic market interpretation

## Forward-Looking Outlook
- Strategic investment outlook by portfolio
- Projected growth analysis
- CIO-style portfolio commentary
- Market environment fit analysis
- Portfolio-specific strategic interpretation

## Investor Profile Tool
Interactive investor suitability questionnaire used to estimate which portfolio strategy best aligns with an investor’s:
- risk tolerance
- investment horizon
- income needs
- long-term objectives

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yFinance
- SciPy
- Requests
- Alpha Vantage API

---

# Portfolio Methodology

The portfolio construction process uses Monte Carlo simulation to generate thousands of potential portfolios under allocation constraints tailored to each investor mandate.

The portfolios are evaluated using:
- expected return
- volatility
- Sharpe ratio
- diversification characteristics
- drawdown behavior

The framework is designed to emphasize:
- diversification
- risk-adjusted return
- long-term strategic positioning
- portfolio suitability across different investor profiles

---

# Project Structure

```text
app.py
README.md
requirements.txt
.env


---

# How to Run the Application

## 1. Clone the Repository

```bash
git clone <your_repo_url>
cd <repo_name>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Add API Key

Create a `.env` file and include:

```text
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## 4. Run the Application

```bash
streamlit run app.py
```

---

# Disclaimer

This project is for educational and demonstration purposes only and does not constitute investment advice, financial planning advice, or a recommendation to buy or sell any security.

Projected returns, forward-looking commentary, and portfolio allocations are hypothetical and based on simplified assumptions.

Past performance does not guarantee future results.

---

# Author

Daniel Pelerman

Master of Business Analytics  
Cal Poly San Luis Obispo