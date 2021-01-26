from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod
from torch import Tensor

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Agent(ABC):

    @staticmethod
    def initialise(params: Dict=None) -> Agent:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, X: Tensor, y: Tensor, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, x: Tensor) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError