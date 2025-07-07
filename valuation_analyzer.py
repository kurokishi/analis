import numpy as np

class ValuationAnalyzer:
    def __init__(self, fmp_client):
        self.fmp = fmp_client

    def calculate_score(self, ticker):
        data = self.fmp.get_full_fundamentals(ticker)
        ratios = data['ratios']
        profile = data['profile']

        try:
            per = ratios.get('priceEarningsRatio', 0)
            pbv = ratios.get('priceToBookRatio', 0)
            roe = ratios.get('returnOnEquity', 0) * 100
            npm = ratios.get('netProfitMargin', 0) * 100
            dividend_yield = ratios.get('dividendYield', 0) * 100

            score = 0
            score += self._score_per(per)
            score += self._score_pbv(pbv)
            score += self._score_roe(roe)
            score += self._score_npm(npm)
            score += self._score_dividend(dividend_yield)
            return score
        except:
            return 0

    def _score_per(self, per):
        if per <= 0: return 0
        if per < 15: return 3
        if per < 20: return 2
        if per < 25: return 1
        return 0

    def _score_pbv(self, pbv):
        if pbv <= 0: return 0
        if pbv < 1: return 3
        if pbv < 1.5: return 2
        if pbv < 2: return 1
        return 0

    def _score_roe(self, roe):
        if roe > 20: return 3
        if roe > 15: return 2
        if roe > 10: return 1
        return 0

    def _score_npm(self, npm):
        if npm > 20: return 3
        if npm > 15: return 2
        if npm > 10: return 1
        return 0

    def _score_dividend(self, dy):
        if dy > 5: return 3
        if dy > 3: return 2
        if dy > 1: return 1
        return 0

    def simulate_allocation(self, df, investment_amount):
        tickers = df['Ticker'].apply(lambda t: t.replace('.JK', ''))
        scores = [self.calculate_score(tkr) for tkr in tickers]
        df['Valuation Score'] = scores
        total_score = sum(scores)

        if total_score > 0:
            df['Allocation Weight'] = df['Valuation Score'] / total_score
        else:
            df['Allocation Weight'] = 1 / len(df)

        df['Allocation Amount'] = df['Allocation Weight'] * investment_amount
        df['Additional Shares'] = (df['Allocation Amount'] / df['Current Price']).astype(int)
        df['Additional Investment'] = df['Additional Shares'] * df['Current Price']

        return df
