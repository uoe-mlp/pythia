from __future__ import annotations
from typing import List, Union, Dict, Tuple
import pandas as pd
from pandas._libs.tslibs import Timestamp
import numpy as np
import os
from functools import reduce

from pythia.journal import TradeOrder, TradeFill, TradeOrderSell, TradeOrderBuy
from pythia.utils import ArgsParser

from .market import Market


class DailyHistoricalMarket(Market):

    def __init__(self, X: np.ndarray, Y: np.ndarray, timestamps: List[pd.Timestamp], trading_cost: float, features_paths: List[str], target_paths: List[str]):
        super(DailyHistoricalMarket, self).__init__(X.shape[1], Y.shape[1], timestamps)
        self.X: np.ndarray= X
        self.Y: np.ndarray = Y
        self.trading_cost: float = trading_cost
        self.features_paths: List[str] = features_paths
        self.target_paths: List[str] = target_paths
        self.input_size = self.X.shape[1]
        self.output_size = self.Y.shape[1]

    @staticmethod
    def initialise(params: Dict) -> Market:
        features_raw: Union[List[str], str] = ArgsParser.get_or_error(params, 'features')
        features: List[str] = [features_raw] if isinstance(features_raw, str) else features_raw
        f_df_arr: List[pd.DataFrame] = []
        for feature in features:
            f_df_tmp = pd.read_csv(os.path.join('data', 'markets', feature))
            f_df_tmp.set_index(pd.DatetimeIndex(f_df_tmp['date']), inplace=True)
            f_df_tmp.drop('date', axis=1, inplace=True)
            f_df_arr.append(f_df_tmp)
        
        targets_raw: Union[List[str], str] = ArgsParser.get_or_error(params, 'targets')
        targets: List[str] = [targets_raw] if isinstance(targets_raw, str) else targets_raw
        t_df_arr: List[pd.DataFrame] = []
        for target in targets:
            t_df_tmp = pd.read_csv(os.path.join('data', 'markets', target))
            t_df_tmp.set_index(pd.DatetimeIndex(t_df_tmp['date']), inplace=True)
            t_df_tmp.drop('date', axis=1, inplace=True)
            t_df_arr.append(t_df_tmp)

        X, Y, dates = DailyHistoricalMarket.combine_datasets(f_df_arr, t_df_arr)

        # Check to see if we should use the previous adjusted close as a parameter
        if ArgsParser.get_or_default(params, 'close_as_feature', False):
            dates = dates[1:]
            X = np.concatenate((X[1:,:], Y[:-1,:]), axis=-1)
            Y = Y[1:,:]

        trading_cost: float = ArgsParser.get_or_default(params, 'trading_cost', 1e-3)

        return DailyHistoricalMarket(X, Y, dates, trading_cost, features, targets)

    def execute(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]:
        # We do not know market impact, the only way to know it is to simulate
        return self.simulate(trades, timestamp)

    def simulate(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]:
        idx = sum([timestamp > x for x in self.timestamps])
        prices = self.Y[idx, :]
        
        # Step 1: Sell
        fills: List[TradeFill] = []
        sell_value: float = 0.0
        for trade in trades:
            if isinstance(trade, TradeOrderSell):
                if trade.instrument > 0:
                    value = trade.quantity * float(prices[trade.instrument]) * (1 - self.trading_cost)
                    sell_value += value
                    fills.append(TradeFill(trade.instrument, trade.started, float(value), float(trade.quantity), timestamp, float(prices[trade.instrument]) * (1 - self.trading_cost), 'sell', trade.id))
                else:
                    value = trade.quantity * float(prices[trade.instrument])
                    sell_value += value
                    fills.append(TradeFill(trade.instrument, trade.started, float(value), float(trade.quantity), timestamp, float(prices[trade.instrument]), 'sell', trade.id))
            elif isinstance(trade, TradeOrderBuy):
                pass
            else:
                raise ValueError('Type of order not recognized')
        
        # Step 2: Buy
        for trade in trades:
            if isinstance(trade, TradeOrderSell):
                pass
            elif isinstance(trade, TradeOrderBuy):
                if trade.instrument > 0:
                    value = trade.percentage * sell_value
                    price = float(prices[trade.instrument]) * (1 + self.trading_cost)
                    fills.append(TradeFill(trade.instrument, trade.started, float(value), float(value / price), timestamp, float(price), 'buy', trade.id))
                else:
                    value = trade.percentage * sell_value
                    price = float(prices[trade.instrument])
                    fills.append(TradeFill(trade.instrument, trade.started, float(value), float(value / price), timestamp, float(price), 'buy', trade.id))    
            else:
                raise ValueError('Type of order not recognized')

        return fills

    @staticmethod
    def combine_datasets(features: List[pd.DataFrame], targets: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        targets_df = reduce(lambda left,right: pd.merge(left,right,on='date'), targets)
        targets_df.sort_index(inplace=True)
        targets_df.bfill(inplace=True)
        targets_df = pd.concat([pd.Series([1 for x in targets_df.index.tolist()], index=targets_df.index.tolist()), targets_df], axis=1)

        for feature in features:
            feature = feature.reindex(targets_df.index)
            feature.ffill(inplace=True)

        features_df = reduce(lambda left,right: pd.merge(left,right,on='date'), features)

        return (np.array(features_df.values), np.array(targets_df.values), [pd.Timestamp(x) for x in targets_df.index])