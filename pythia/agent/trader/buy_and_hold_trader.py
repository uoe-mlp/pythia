from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty
import numpy as np
from pandas import Timestamp

from pythia.journal import TradeOrderSell, TradeOrderBuy, TradeOrder
from pythia.journal import TradeFill
from pythia.utils import ArgsParser

from .trader import Trader


class BuyAndHoldTrader(Trader):

    def __init__(self, output_size: int):
        super(BuyAndHoldTrader, self).__init__(output_size=output_size)

    @staticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        return BuyAndHoldTrader(output_size)

    def act(self, prediction: np.ndarray, conviction: np.ndarray, timestamp: Timestamp, prices: np.ndarray, predict_returns: bool) -> List[TradeOrder]:
        # Implicitly checking if we are in the first period
        trades: List[TradeOrder] = []
        if self.portfolio[0] == 1:
            trades.append(TradeOrderSell(0, timestamp, self._portfolio[0]))

            tot_assets = self._portfolio.shape[0] - 1
            for i in range(tot_assets):
                trades.append(TradeOrderBuy(i + 1, timestamp, 1.0 / tot_assets))
        return trades