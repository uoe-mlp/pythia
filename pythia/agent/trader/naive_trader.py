from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty
import numpy as np
from pandas import Timestamp

from pythia.journal import TradeOrderSell, TradeOrderBuy, TradeOrder
from pythia.journal import TradeFill
from pythia.utils import ArgsParser

from .trader import Trader


class NaiveTrader(Trader):

    def __init__(self, output_size: int):
        super(NaiveTrader, self).__init__(output_size=output_size)

    @staticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        return NaiveTrader(output_size)

    def act(self, prediction: np.ndarray, conviction: np.ndarray, timestamp: Timestamp, prices: np.ndarray, predict_returns: bool) -> List[TradeOrder]:
        if predict_returns:
            target_trade = int(np.argmax(prediction))
        else:
            target_trade = int(np.argmax(prediction / prices))

        target_portfolio: np.ndarray = self._portfolio * 0
        target_portfolio[target_trade] = 1
        current_portfolio: np.ndarray = self.portfolio * prices
        current_portfolio /= sum(current_portfolio)
        delta = target_portfolio - current_portfolio
        
        trades: List[TradeOrder] = [TradeOrderSell(i, timestamp, x) for i, x in enumerate(self._portfolio) if float(delta[i]) < 0]
        trades += [TradeOrderBuy(target_trade, timestamp, percentage=1) for i, x in enumerate(self._portfolio) if float(delta[i]) > 0]
        return trades
        