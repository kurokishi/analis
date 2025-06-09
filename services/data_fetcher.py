# services/data_fetcher.py
import yfinance as yf
from datetime import datetime, timedelta

def get_price_data(ticker, days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df.dropna()
    return df

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "market_cap": info.get("marketCap"),
        "dividend_yield": info.get("dividendYield"),
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "eps": info.get("trailingEps"),
        "revenue": info.get("totalRevenue"),
        "net_income": info.get("netIncomeToCommon"),
        "revenue_growth": info.get("revenueGrowth"),
        "net_margin": info.get("netMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "interest_coverage": info.get("ebitda") / info.get("interestExpense", 1) if info.get("interestExpense") else None,
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "last_dividend": info.get("lastDividendValue"),
        "dividend_payout_ratio": info.get("payoutRatio"),
        "dividend_history": [
            {"year": 2023, "amount": info.get("lastDividendValue")},
            {"year": 2022, "amount": info.get("lastDividendValue")},
            {"year": 2021, "amount": info.get("lastDividendValue")}
        ]
    }
