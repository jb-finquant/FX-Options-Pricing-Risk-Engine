import yfinance as yf
import numpy as np


def load_market_data(ticker: str = "EURUSD=X", period: str = "1y"):
    """
    Downloads historical market data and computes basic model inputs.

    Retrieves adjusted closing prices from Yahoo Finance and derives
    the current spot level, annualized historical volatility, and
    daily log returns for use in pricing and risk analysis.

    Notes
    -----
    Historical volatility is computed assuming 252 trading days
    per year and is used as a proxy for model volatility.

    Raises
    ------
    ConnectionError: If market data cannot be retrieved from Yahoo Finance.

    ValueError: If insufficient data is returned for the given ticker/period.
    """
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise ConnectionError(f"Failed to download market data for {ticker}: {e}")

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' with period '{period}'.")

    closes = data["Close"].squeeze().dropna()

    if len(closes) < 30:
        raise ValueError(
            f"Insufficient data for '{ticker}': {len(closes)} observations. "
            f"At least 30 required for reliable volatility estimation."
        )

    S0        = float(closes.iloc[-1])
    date_from = closes.index[0].strftime("%Y-%m-%d")
    date_to   = closes.index[-1].strftime("%Y-%m-%d")

    returns = np.log(closes / closes.shift(1)).dropna().values
    vol     = float(np.std(returns) * np.sqrt(252))

    return S0, vol, returns, date_from, date_to