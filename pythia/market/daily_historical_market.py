from __future__ import annotations
from typing import List, Union, Dict, Tuple
import pandas as pd
from pandas._libs.tslibs import Timestamp
from torch import Tensor, cat
import os
from functools import reduce

from pythia.journal import TradeOrder, TradeFill, TradeOrderSell, TradeOrderBuy
from pythia.utils import ArgsParser

from .market import Market


class DailyHistoricalMarket(Market):

    def __init__(self, X: Tensor, Y: Tensor, timestamps: List[pd.Timestamp], trading_cost: float, features_paths: List[str], target_paths: List[str]):
        super(DailyHistoricalMarket, self).__init__(X.shape[1], Y.shape[1], timestamps)
        self.X: Tensor= X
        self.Y: Tensor = Y
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

        trading_cost: float = ArgsParser.get_or_default(params, 'trading_cost', 1e-3)               # TODO: add fixed+linear as an option

        return DailyHistoricalMarket(X, Y, dates, trading_cost, features, targets)

    def execute(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]:
        # We do not know market impact, the only way to know it is to simulate
        return self.simulate(trades, timestamp)

    def simulate(self, trades: List[TradeOrder], timestamp: Timestamp) -> List[TradeFill]:
        # Step 1: evaluate things to sell
        sell_orders = [x for x in trades if isinstance(x, TradeOrderSell)]
        idx = sum([timestamp > x for x in self.timestamps])
        prices = cat((Tensor([1]), self.Y[idx, :]), 0)  # Adding cash
        value = sum([x.quantity * prices[x.instrument] for x in sell_orders])

        # Step 2: net based on this valuation
        orders: Tensor = prices * 0
        for trade in trades:
            if isinstance(trade, TradeOrderSell):
                orders[trade.instrument] = orders[trade.instrument] - trade.quantity
            elif isinstance(trade, TradeOrderBuy):
                orders[trade.instrument] = orders[trade.instrument] + (trade.percentage * value / prices[trade.instrument])
            else:
                raise ValueError('Type of order not recognized')
        
        fills = {}

        # If net negative, sell
        sell_value: float = 0.0
        for i, order in enumerate(orders):
            if order < 0:
                sell_value += order * prices[i] * (1 - self.trading_cost)
                orders[i] = 0
                fills[i] = (order, prices[i] * (1 - self.trading_cost))

        # With proceedings, buy longs
        orders = orders * prices / sum(orders * prices) * sell_value / (1 + self.trading_cost)
        raise NotImplementedError
        

        # Step 4: return remaining trades

        return []

    @staticmethod
    def combine_datasets(features: List[pd.DataFrame], targets: List[pd.DataFrame]) -> Tuple[Tensor, Tensor, List[pd.Timestamp]]:
        targets_df = reduce(lambda left,right: pd.merge(left,right,on='date'), targets)
        targets_df.sort_index(inplace=True)
        targets_df.bfill(inplace=True)

        for feature in features:
            feature = feature.reindex(targets_df.index)
            feature.ffill(inplace=True)

        features_df = reduce(lambda left,right: pd.merge(left,right,on='date'), features)

        return (Tensor(features_df.values), Tensor(targets_df.values), [pd.Timestamp(x) for x in targets_df.index])