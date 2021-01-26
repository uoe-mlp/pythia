from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty
from torch import Tensor

from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from .position import Position

class Trader(ABC):

    @abstractproperty
    def portfolio(self) -> List[Position]:
        raise NotImplementedError

    @staticmethod
    def initialise(params: Dict=None) -> Trader:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, prediction: Tensor, conviction: Tensor, y: Tensor, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, prediction: Tensor, conviction: Tensor) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError