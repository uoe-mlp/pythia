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
        self._output_size: int = output_size
        self._portfolio: np.array = np.eye(1, output_size)[0] # Fully invested in the first asset initially (cash index must be 0)

    @property
    def portfolio(self) -> np.array:
        return self._portfolio

    @staticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        return NaiveTrader(output_size)

    def fit(self, prediction: np.array, conviction: np.array, Y: np.array, **kwargs):
        pass

    def act(self, prediction: np.array, conviction: np.array, timestamp: Timestamp, prices: np.array, predict_returns: bool) -> List[TradeOrder]:
        if predict_returns:
            target_trade = int(np.argmax(prediction))
        else:
            target_trade = int(np.argmax(prediction / prices))

        target_portfolio: np.array = self._portfolio * 0
        target_portfolio[target_trade] = 1
        current_portfolio: np.array = self.portfolio * prices
        current_portfolio /= sum(current_portfolio)
        delta = target_portfolio - current_portfolio
        
        trades: List[TradeOrder] = [TradeOrderSell(i, timestamp, x) for i, x in enumerate(self._portfolio) if float(delta[i]) < 0]
        trades += [TradeOrderBuy(target_trade, timestamp, percentage=1) for i, x in enumerate(self._portfolio) if float(delta[i]) > 0]
        return trades

    def update_portfolio(self, fills: List[TradeFill]):
        for fill in fills:
            if fill.direction == 'buy':
                self._portfolio[fill.instrument] += fill.quantity
            elif fill.direction == 'sell':
                self._portfolio[fill.instrument] -= fill.quantity
            else:
                raise ValueError('Direction not recognized.')
        