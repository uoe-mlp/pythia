from __future__ import annotations
from typing import Dict, Optional, List
from abc import ABC, abstractclassmethod
from torch import Tensor
from pandas import Timestamp

from pythia.journal import TradeOrder
from pythia.journal import TradeFill

from .agent import Agent
from .predictor import Predictor
from .trader import Trader


class SupervisedAgent(Agent):

    def __init__(self, predictor: Predictor, trader: Trader):
        self.predictor: Predictor = predictor
        self.trader: Trader = trader

    @staticmethod
    def initialise(params: Dict=None) -> SupervisedAgent:
        raise NotImplementedError

    def fit(self, X: Tensor, y: Tensor, **kwargs):
        self.predictor.fit(X, y, **kwargs)
        prediction, conviction = self.predictor.predict(X)
        self.trader.fit(prediction=prediction, conviction=conviction, y=y)

    def act(self, x: Tensor, timestamp: Timestamp, prices: Tensor) -> List[TradeOrder]:
        prediction, conviction = self.predictor.predict(x)
        return self.trader.act(prediction, conviction, timestamp, prices)

    def update_portfolio(self, fills: List[TradeFill]):
        self.trader.update_portfolio(fills)