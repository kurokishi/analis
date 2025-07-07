import requests

class FMPClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def _get(self, endpoint):
        try:
            url = f"{self.base_url}/{endpoint}&apikey={self.api_key}" if "?" in endpoint else f"{self.base_url}/{endpoint}?apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching data from FMP: {e}")
            return None

    def get_profile(self, ticker):
        data = self._get(f"profile/{ticker}")
        return data[0] if data else {}

    def get_ratios(self, ticker):
        data = self._get(f"ratios/{ticker}?period=annual")
        return data[0] if data else {}

    def get_cashflow(self, ticker):
        data = self._get(f"cash-flow-statement/{ticker}?period=annual")
        return data[0] if data else {}

    def get_quote(self, ticker):
        data = self._get(f"quote/{ticker}")
        return data[0] if data else {}

    def get_growth(self, ticker):
        data = self._get(f"income-statement-growth/{ticker}?period=annual")
        return data[0] if data else {}

    def get_sector_performance(self):
        return self._get("sector-performance")

    def get_full_fundamentals(self, ticker):
        return {
            'profile': self.get_profile(ticker),
            'ratios': self.get_ratios(ticker),
            'cashflow': self.get_cashflow(ticker),
            'quote': self.get_quote(ticker),
            'growth': self.get_growth(ticker)
        }
