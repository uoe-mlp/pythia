from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty
from torch import Tensor, argmax, eye
from pandas import Timestamp

from pythia.journal import TradeOrderSell, TradeOrderBuy, TradeOrder
from pythia.journal import TradeFill
from pythia.utils import ArgsParser

from .trader import Trader


class NaiveTrader(Trader):

    def __init__(self, output_size: int):
        self._output_size: int = output_size
        self._portfolio: Tensor = eye(1, output_size)[0] # Fully invested in the first asset initially (cash index must be 0)

    @property
    def portfolio(self) -> Tensor:
        return self._portfolio

    @staticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        return NaiveTrader(output_size)

    def fit(self, prediction: Tensor, conviction: Tensor, Y: Tensor, **kwargs):
        pass

    def act(self, prediction: Tensor, conviction: Tensor, timestamp: Timestamp, prices: Tensor, returns: bool) -> List[TradeOrder]:
        if returns:
            target_trade = int(argmax(prediction))
        else:
            target_trade = int(argmax(prediction / prices))

        target_portfolio: Tensor = self._portfolio * 0
        target_portfolio[target_trade] = 1
        current_portfolio: Tensor = self.portfolio * prices
        current_portfolio /= sum(current_portfolio)
        delta = target_portfolio - current_portfolio
        
        trades: List[TradeOrder] = [TradeOrderSell(i, timestamp, x) for i, x in enumerate(self._portfolio) if float(delta[i]) < 0]
        trades += [TradeOrderBuy(target_trade, timestamp, percentage=1) for i, x in enumerate(self._portfolio) if float(delta[i]) > 0]
        return trades

    def update_portfolio(self, fills: List[TradeFill]):
        for fill in fills:
            if fill.direction == 'buy':
                self._portfolio[fill.instrument] += fill.quantity
                self._portfolio[0] -= fill.value
            elif fill.direction == 'sell':
                self._portfolio[fill.instrument] -= fill.quantity
                self._portfolio[0] += fill.value
            else:
                raise ValueError('Direction not recognized.')
        