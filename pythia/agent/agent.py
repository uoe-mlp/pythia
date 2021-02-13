from __future__ import annotations
from typing import Dict, List, Callable
from abc import ABC, abstractclassmethod
import numpy as np
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Agent(ABC):

    @staticmethod
    def initialise(input_size: int, output_size: int, params: Dict) -> Agent:
        pass

    @abstractclassmethod
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, simulator: Callable[[List[TradeOrder], Timestamp], List[TradeFill]], **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, X: np.ndarray, timestamp: Timestamp, Y: np.ndarray) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError
