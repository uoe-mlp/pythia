from __future__ import annotations
from datetime import time
from pandas._libs.tslibs import Timestamp
import numpy as np
from typing import List, Dict, Any
import pandas as pd

from .trade_fill import TradeFill


class Analytics(object):

    def __init__(self, timeseries: pd.Series, volatility: float, cumulative_return: float, sharpe_ratio: float, sortino_ratio: float, maximum_drawdown: float):
        self.timeseries: pd.Series = timeseries
        self.volatility: float = volatility
        self.cumulative_return: float = cumulative_return
        self.sharpe_ratio: float = sharpe_ratio
        self.sortino_ratio: float = sortino_ratio
        self.maximum_drawdown: float = maximum_drawdown

    @staticmethod
    def initialise(timestamps: List[pd.Timestamp], fills: List[TradeFill], prices: np.ndarray) -> Analytics:
        holdings_df = pd.DataFrame(prices * 0, index=timestamps).astype('float')
        holdings_df.iloc[0, 0] = 1      # Initially, all in the first asset (cash)

        for fill in fills:
            if fill.completed < timestamps[-1]:                 # We exclude last day as trading last minute is obviously a bad decision (pay commissions, but cannot get returns)
                if fill.direction.lower() == 'buy':
                    holdings_df.loc[fill.completed][fill.instrument] += fill.quantity
                elif fill.direction.lower() == 'sell':
                    holdings_df.loc[fill.completed][fill.instrument] -= fill.quantity

        holdings_df = holdings_df.cumsum()
        prices_df = pd.DataFrame(prices, index=timestamps).astype('double')

        timeseries = (prices_df * holdings_df).sum(axis=1)
        timeseries /= timeseries[0]               # We subtract the initial trading cost (aka, we assume we start from an ideal scenario) 
        return Analytics(
            timeseries,
            cumulative_return=timeseries.iloc[-1], 
            volatility=Analytics.calculate_volatility(timeseries),
            sharpe_ratio=Analytics.calculate_sharpe_ratio(timeseries),
            sortino_ratio=Analytics.calculate_sortino_ratio(timeseries),
            maximum_drawdown=Analytics.calculate_maximum_drawdonw(timeseries))

    @staticmethod
    def timeseries2returns(timeseries: pd.Series) -> pd.Series:
        ret = (timeseries / timeseries.shift(1)) - 1
        ret.dropna(inplace=True)
        return ret

    @staticmethod
    def calculate_volatility(timeseries: pd.Series) -> float:
        return Analytics.timeseries2returns(timeseries).std()

    @staticmethod
    def calculate_sharpe_ratio(timeseries: pd.Series) -> float:
        ret = Analytics.timeseries2returns(timeseries)
        return ret.mean() / ret.std()
    
    @staticmethod
    def calculate_sortino_ratio(timeseries: pd.Series) -> float:
        return Analytics.timeseries2returns(timeseries).mean() / Analytics.calculate_donwside_deviation(timeseries)

    @staticmethod
    def calculate_donwside_deviation(timeseries: pd.Series) -> float:
        ret = Analytics.timeseries2returns(timeseries)
        return (((ret).map(lambda x: min(x, 0)) ** 2).sum() / (ret.shape[0] - 1)) ** 0.5

    @staticmethod
    def calculate_maximum_drawdonw(timeseries: pd.Series) -> float:
        return timeseries.div(timeseries.cummax()).sub(1).min()

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        data['timeseries'] = {
            'values': self.timeseries.to_list(), 
            'dates': [x.strftime('%Y-%m-%d') for x in self.timeseries.index.tolist()]}
        data['volatility'] = self.volatility
        data['cumulative_return'] = self.cumulative_return
        data['sharpe_ratio'] = self.sharpe_ratio
        data['sortino_ratio'] = self.sortino_ratio
        data['maximum_drawdown'] = self.maximum_drawdown

        return data