from __future__ import annotations
from typing import Dict, List
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty, abstractstaticmethod
import numpy as np
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill


class Trader(ABC):

    def __init__(self, output_size: int):
        self._output_size: int = output_size
        self._portfolio: np.ndarray = Trader.get_default_portfolio(output_size)

    @property
    def portfolio(self) -> np.ndarray:
        return self._portfolio

    @property
    def output_size(self) -> int:
        return self._output_size

    @staticmethod
    @abstractstaticmethod
    def initialise(output_size: int, params: Dict) -> Trader:
        raise NotImplementedError

    def fit(self, prediction: np.ndarray, conviction: np.ndarray, Y: np.ndarray, predict_returns: bool, **kwargs):pass

    @abstractclassmethod
    def act(self, prediction: np.ndarray, conviction: np.ndarray, timestamp: Timestamp, prices: np.ndarray, predict_returns: bool) -> List[TradeOrder]:
        raise NotImplementedError

    def update_portfolio(self, fills: List[TradeFill]):
        for fill in fills:
            if fill.direction == 'buy':
                self._portfolio[fill.instrument] += fill.quantity
            elif fill.direction == 'sell':
                self._portfolio[fill.instrument] -= fill.quantity
            else:
                raise ValueError('Direction not recognized.')

    def update_policy(self, X: np.ndarray, Y: np.ndarray, prediction: np.ndarray, conviction: np.ndarray, predict_returns: bool) -> None: pass

    @staticmethod
    def get_default_portfolio(output_size: int) -> np.ndarray:
        return np.eye(1, output_size)[0] # Fully invested in the first asset initially (cash index must be 0)

    def clean_portfolio(self) -> None:
        self._portfolio = Trader.get_default_portfolio(self._output_size)