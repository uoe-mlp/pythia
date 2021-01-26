from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty
from torch import Tensor
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from .position import Position

class Trader(ABC):

    @abstractproperty
    def portfolio(self) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def initialise(params: Dict) -> Trader:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, prediction: Tensor, conviction: Tensor, y: Tensor, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, prediction: Tensor, conviction: Tensor, timestamp: Timestamp, prices: Tensor) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError

    def calculate_value(self, prices: Tensor, positions: List[Position]) -> float:
        return sum([x.quantity * float(prices[x.instrument]) for x in positions])