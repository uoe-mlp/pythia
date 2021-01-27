from __future__ import annotations
from typing import List, Union, Dict, Tuple
import pandas as pd
from torch import Tensor
import os
from functools import reduce

from pythia.journal import TradeOrder, TradeFill
from pythia.utils import ArgsParser

from .market import Market


class DailyHistoricalMarket(Market):

    def __init__(self, X: Tensor, Y: Tensor, dates: List[pd.Timestamp], trading_cost: float, features_paths: List[str], target_path: str):
        super(DailyHistoricalMarket, self).__init__(X.shape[1], Y.shape[1])
        self.X: Tensor= X
        self.Y: Tensor = Y
        self.dates: List[pd.Timestamp] = dates
        self.trading_cost: float = trading_cost
        self.features_paths: List[str] = features_paths
        self.target_path: str = target_path

    @staticmethod
    def initialise(params: Dict) -> Market:
        features_raw: Union[List[str], str] = ArgsParser.get_or_error(params, 'features')
        features: List[str] = [features_raw] if isinstance(features_raw, str) else features_raw
        f_df_arr: List[pd.DataFrame] = []
        for feature in features:
            t_df_tmp = pd.read_csv(os.path.join('data', 'markets', feature))
            t_df_tmp.set_index(pd.DatetimeIndex(t_df_tmp['date']), inplace=True)
            t_df_tmp.drop('date', axis=1, inplace=True)
            f_df_arr.append(t_df_tmp)
        
        target: str = ArgsParser.get_or_error(params, 'target')
        t_df: pd.DataFrame = pd.read_csv(os.path.join('data', 'markets', target))
        t_df.set_index(pd.DatetimeIndex(t_df['date']), inplace=True)
        t_df.drop('date', axis=1, inplace=True)

        X, Y, dates = DailyHistoricalMarket.combine_datasets(f_df_arr, t_df)

        trading_cost: float = ArgsParser.get_or_default(params, 'trading_cost', 1e-3)               # TODO: add fixed+linear as an option

        return DailyHistoricalMarket(X, Y, dates, trading_cost, features, target)

    def execute(self, trades: List[TradeOrder]) -> List[TradeFill]:
        raise NotImplementedError

    def get_prices(self, timestamp: pd.Timestamp) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def combine_datasets(features: List[pd.DataFrame], target: pd.DataFrame) -> Tuple[Tensor, Tensor, List[pd.Timestamp]]:
        target.sort_index(inplace=True)
        target.bfill(inplace=True)

        for feature in features:
            feature = feature.reindex(target.index)
            feature.ffill(inplace=True)

        features_df = reduce(lambda left,right: pd.merge(left,right,on='date'), features)

        return (Tensor(features_df.values), Tensor(target.values), [pd.Timestamp(x) for x in target.index])