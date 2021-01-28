from __future__ import annotations
from typing import Dict, List, Callable
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
    def fit(self, X_train: Tensor, Y_train: Tensor, X_val: Tensor, Y_val: Tensor, simulator: Callable[[List[TradeOrder], Timestamp], List[TradeFill]], **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, x: Tensor, timestamp: Timestamp) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError