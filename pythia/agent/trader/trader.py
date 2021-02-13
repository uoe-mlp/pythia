from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractproperty, abstractstaticmethod
import numpy as np
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Trader(ABC):

    @abstractproperty
    def portfolio(self) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractstaticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        raise NotImplementedError

    @abstractclassmethod
    def fit(self, prediction: np.ndarray, conviction: np.ndarray, Y: np.ndarray, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def act(self, prediction: np.ndarray, conviction: np.ndarray, timestamp: Timestamp, prices: np.ndarray, predict_returns: bool) -> List[TradeOrder]:
        raise NotImplementedError

    @abstractclassmethod
    def update_portfolio(self, fills: List[TradeFill]):
        raise NotImplementedError
