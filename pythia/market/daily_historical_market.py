from __future__ import annotations
from typing import List, Union, Dict, Tuple
from pandas import Timestamp, DataFrame, read_csv
from torch import Tensor
import os

from pythia.journal import TradeOrder, TradeFill
from pythia.utils import ArgsParser

from .market import Market


class DailyHistoricalMarket(Market):

    def __init__(self, X: Tensor, Y: Tensor, dates: List[Timestamp], trading_cost: float, features_paths: List[str], target_path: str):
        super(DailyHistoricalMarket, self).__init__(X.shape[1], Y.shape[1])
        self.X: Tensor= X
        self.Y: Tensor = Y
        self.dates: List[Timestamp] = dates
        self.trading_cost: float = trading_cost
        self.features_paths: List[str] = features_paths
        self.target_path: str = target_path

    @staticmethod
    def initialise(params: Dict) -> Market:
        features_raw: Union[List[str], str] = ArgsParser.get_or_error(params, 'features')
        features: List[str] = [features_raw] if isinstance(features_raw, str) else features_raw
        f_df_arr: List[DataFrame] = []
        for feature in features:
            f_df_arr.append(read_csv(os.path.join('data', 'markets', feature)))
        
        target: str = ArgsParser.get_or_error(params, 'targets')
        t_df: DataFrame = read_csv(os.path.join('data', 'markets', target))

        X, Y, dates = DailyHistoricalMarket.combine_datasets(f_df_arr, t_df)

        trading_cost: float = ArgsParser.get_or_default(params, 'trading_cost', 1e-3)               # TODO: add fixed+linear as an option

        return DailyHistoricalMarket(X, Y, dates, trading_cost, features, target)

    def execute(self, trades: List[TradeOrder]) -> List[TradeFill]:
        raise NotImplementedError

    def get_prices(self, timestamp: Timestamp) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def combine_datasets(features: List[DataFrame], target: DataFrame) -> Tuple[Tensor, Tensor, List[Timestamp]]:
        # TODO: bfill target (worst case scenario)
        # TODO: reindex features on dates from target, with ffill of info

        return (Tensor(), Tensor(), [Timestamp()])