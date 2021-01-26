from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty
from torch import Tensor, argmax, eye
from pandas import Timestamp

from pythia.journal import TradeOrderSell, TradeOrderBuy, TradeOrder
from pythia.journal import TradeFill
from pythia.utils import ArgsParser

from .position import Position
from .trader import Trader


class NaiveTrader(Trader):

    def __init__(self, output_size: int):
        self._output_size: int = output_size
        self._portfolio: Tensor = eye(output_size, 1) # Fully invested in cash initially (cash index must be 0)

    def portfolio(self) -> Tensor:
        return self._portfolio

    @staticmethod
    def initialise(params: Dict) -> Trader:
        output_size: int = ArgsParser.get_or_error(params, 'output_size')
        return NaiveTrader(output_size)

    def fit(self, prediction: Tensor, conviction: Tensor, y: Tensor, **kwargs):
        pass

    def act(self, prediction: Tensor, conviction: Tensor, timestamp: Timestamp, prices: Tensor) -> List[TradeOrder]:
        target_trade = int(argmax(prediction))
        trades: List[TradeOrder] = [TradeOrderSell(i, timestamp, x) for i, x in enumerate(self._portfolio)]
        trades.append(TradeOrderBuy(target_trade, timestamp, percentage=1))
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
        