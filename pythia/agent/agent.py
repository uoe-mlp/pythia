from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod
from torch import Tensor
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Agent(ABC):

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Agent:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, X: Tensor, Y: Tensor, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, x: Tensor, timestamp: Timestamp, prices: Tensor) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError